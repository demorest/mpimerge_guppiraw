/* mpimerge_guppiraw.c
 *
 * MPI-based file merging program for raw guppi
 * data (either TDOM or channelized baseband).
 *
 * P. Demorest, Feb 2013
 */
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include "mpi.h"
#include "fitshead.h"

#define MAX_HEADER_SIZE (2880*64)

void usage() {
    printf("usage:  mpimerge_guppiraw [options] basefilename\n"
           "Options:\n"
           "  -o name, --output=name   Output base filename (auto-generate)\n"
           "  -i nn, --initial=nn      Starting input file number (1)\n"
           "  -f nn, --final=nn        Ending input file number (auto)\n"
           "\n");
}

int read_header(FILE *f, char *hdr);
void tdom_reorder(char *out, char *in, int nnode, 
        int pktsize_bytes, int blocsize_node);

int main(int argc, char *argv[])
{
    int i, ii, status = 0, statsum = 0;
    int fnum_start = 1, fnum_end = 0;
    int numprocs, numbands, myid;
    char hostname[256];
    char output_base[256] = "\0";
    MPI_Status mpistat;
    /* Cmd line */
    static struct option long_opts[] = {
        {"output",  1, NULL, 'o'},
        {"initial", 1, NULL, 'i'},
        {"final",   1, NULL, 'f'},
        {0,0,0,0}
    };
    int opt, opti;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    numbands = numprocs - 1;

    // Process the command line
    while ((opt=getopt_long(argc,argv,"o:i:f:",long_opts,&opti))!=-1) {
        switch (opt) {
        case 'o':
            strncpy(output_base, optarg, 255);
            output_base[255]='\0';
            break;
        case 'i':
            fnum_start = atoi(optarg);
            break;
        case 'f':
            fnum_end = atoi(optarg);
            break;
        default:
            if (myid==0) usage();
            MPI_Finalize();
            exit(0);
            break;
        }
    }
    if (optind==argc) { 
        if (myid==0) usage();
        MPI_Finalize();
        exit(1);
    }
    
    // Determine the hostnames of the processes
    {
        if (gethostname(hostname, 255) < 0)
            strcpy(hostname, "unknown");

        MPI_Barrier(MPI_COMM_WORLD);
        if (myid == 0) printf("\n");
        fflush(NULL);
        for (ii = 0 ; ii < numprocs ; ii++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (myid == ii)
                printf("Process %3d is on machine %s\n", myid, hostname);
            fflush(NULL);
            MPI_Barrier(MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        fflush(NULL);
    }
    
    // Basefilenames for the GPU nodes
    //char basefilename[256];
    //if (myid > 0) {
    //    sprintf(basefilename, "/data/gpu/partial/%s/%s", 
    //            hostname, argv[optind]);
    //}

    // Open the input/output files
    FILE *rawfile = NULL;
    char filenm[256];
    if (myid > 0) {
        //sprintf(filenm, "%s.0000.raw", basefilename);
        sprintf(filenm, "/data/gpu/partial/%s/%s", hostname, argv[optind]);
        rawfile = fopen(filenm, "r");
    } else {
        sprintf(filenm, "%s", argv[optind]);
        rawfile = fopen(filenm, "w");
    }
    if (rawfile==NULL) {
        fprintf(stderr, "Process %d: Error opening '%s'\n", myid, filenm);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Allocate memory buffers
    char *databuf = NULL, *dataout = NULL;
    char header[MAX_HEADER_SIZE];
    int pktidx0=0, blocsize0=0;
    int *pktidx = (int *)malloc(sizeof(int) * numprocs);
    int *blocsize = (int *)malloc(sizeof(int) * numprocs);
    double rf0=0.0;
    double *rf = (double *)malloc(sizeof(double) * numprocs);

    // For MPI vector gather
    int *counts = (int *)malloc(sizeof(int) * numprocs);
    int *offsets = (int *)malloc(sizeof(int) * numprocs);
    counts[0] = offsets[0] = 0;

    // Now loop over the blocks
    do {
        MPI_Barrier(MPI_COMM_WORLD);

        // Read the current block on each of the slave nodes
        // Get blocsize, pktindex params
        if (myid > 0) {

            status=0;

            int rv = read_header(rawfile, header);
            if (rv<=0) status=1; 

            rv = hgeti4(header, "PKTIDX", &pktidx0);
            if (rv==0) status=1;
            rv = hgeti4(header, "BLOCSIZE", &blocsize0);
            if (rv==0) status=1;

            rv = hgetr8(header, "OBSFREQ", &rf0);
            if (rv==0) status=1;

            databuf = (char *)realloc(databuf, blocsize0);
            rv = fread(databuf, 1, blocsize0, rawfile);
            if (rv<blocsize0) status=1;

        } 
        
        // Combine statuses of all nodes by summing....
        MPI_Allreduce(&status, &statsum, 1, 
                      MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        if (statsum) break;

        // Get the blocsize and pktindex for all nodes
        status = MPI_Gather(&blocsize0, 1, MPI_INT, 
                blocsize, 1, MPI_INT,
                0, MPI_COMM_WORLD);
        status = MPI_Gather(&pktidx0, 1, MPI_INT, 
                pktidx, 1, MPI_INT,
                0, MPI_COMM_WORLD);
        status = MPI_Gather(&rf0, 1, MPI_DOUBLE,
                rf, 1, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

        //printf("id=%d blocsize=%d\n", myid, blocsize0);

        // Check for consistency
        if (myid==0) {
            int ok=1;
            blocsize0 = blocsize[1];
            pktidx0 = pktidx[1];
            rf0 = 0.0;
            for (i=1; i<numprocs; i++) {
                //printf("blocsize%d = %d\n", i, blocsize[i]);
                //printf("pktidx%d = %d\n", i, pktidx[i]);
                if (blocsize[i] != blocsize0) ok=0;
                if (pktidx[i] != pktidx0) ok=0;
                rf0 += rf[i];
            }
            if (ok==0) {
                fprintf(stderr, "Inconsistent BLOCSIZE or PKTIDX\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            rf0 /= (double)numbands;
            printf("blocsize = %d\n", blocsize[0]);
            printf("pktidx = %d\n", pktidx[0]);
            printf("rf=%.5f\n", rf0);
            databuf = (char *)realloc(databuf,(long long)blocsize0*numbands);
        }

        // Get all the data on root node
        for (i=1 ; i<numprocs ; i++) {
            counts[i] = blocsize0;
            offsets[i] = (i-1) * blocsize0;
        }
        status = MPI_Gatherv(databuf, blocsize0, MPI_UNSIGNED_CHAR, 
                             databuf, counts, offsets, MPI_UNSIGNED_CHAR, 
                             0, MPI_COMM_WORLD);

        // Get one of the headers onto root node to use as a base..
        if (myid==1) 
            MPI_Send(header, MAX_HEADER_SIZE, MPI_CHAR, 0, 0, 
                    MPI_COMM_WORLD);
        else if (myid==0) 
            MPI_Recv(header, MAX_HEADER_SIZE, MPI_CHAR, 1, 0, 
                    MPI_COMM_WORLD, &mpistat);

        // Tweak header and output data to root node
        if (myid==0) {
            int itmp,nchan;
            double bw, totbw;

            hgetr8(header,"OBSBW",&bw);
            totbw = bw * (double)numbands;
            hputr8(header,"OBSBW",totbw);

            hgeti4(header,"OBSNCHAN",&nchan);
            if (nchan>1) {
                /* Normal PFB-produced data */
                hputi4(header,"OBSNCHAN",nchan*numbands);
                hputnr8(header,"OBSFREQ",10,rf0);
                dataout = databuf;
            } else {
                /* Time-domain data, needs rearranging */
                dataout = (char *)realloc(dataout,
                        (long long)blocsize0*numbands);
                tdom_reorder(dataout, databuf, numbands, 8192, blocsize0);
                hputnr8(header,"OBSFREQ",10,rf0 + bw/2.0);
                hgeti4(header,"OVERLAP",&itmp);
                hputi4(header,"OVERLAP",itmp*numbands);
                hputr8(header,"CHAN_BW",totbw);
                hputr8(header,"TBIN",0.5/(fabs(totbw)*1e6));
                // TODO: mess with pktidx or pktsize??

            }

            // Pretend we were getting larger packets
            // for various indexing reasons.
            hgeti4(header,"PKTSIZE",&itmp);
            hputi4(header,"PKTSIZE",itmp*numbands);
            hputi4(header,"BLOCSIZE",blocsize0*numbands);

            /* Write the header */
            char *optr;
            char *hend = ksearch(header, "END");
            for (optr=header; optr<=hend; optr+=80) {
                fwrite(optr, 80, 1, rawfile);
            } 


            /* Write the combined data */
            fwrite(dataout, 1, (size_t)blocsize0*numbands, rawfile);

        }

    } while (statsum == 0);

    // Free mem buffers
    if (databuf) free(databuf);
    free(blocsize);
    free(pktidx);
    free(counts);
    free(offsets);
    
    // Close the files and finalize things
    MPI_Finalize();
    exit(0);
}

// Read header lines until END is found, or max len is reached
int read_header(FILE *f, char *hdr) {
    const size_t cs = 80; // FITS-style 80 char cards
    char card[cs+1];
    char end[cs+1];
    memset(end,' ',cs);
    strncpy(end,"END",3);
    end[cs] = '\0';
    card[cs] = '\0';
    const int max_cards = MAX_HEADER_SIZE / cs;
    int count = 0;
    int got_end = 0;
    char *ptr = hdr;

    while (count<max_cards && got_end==0) {

        // Read next one
        int rv = fread(card, 1, cs, f);
        if (rv<cs) return -1;

        if (strncmp(card,end,cs)==0) got_end=1; 

        count++;

        strncpy(ptr, card, cs);
        ptr += cs;
        *ptr = '\0'; 
    }

    if (got_end==0) return 0;

    return count;
}

void tdom_reorder(char *out, char *in, int nnode, 
        int pktsize_bytes, int blocsize_node) {
    // TODO complain if nnode != 8, etc?
    char *nodebuf[8]; 
    const int npkt_node = blocsize_node / pktsize_bytes;
    int i;
    for (i=0; i<nnode; i++) 
        nodebuf[i] = in + (long long)i*blocsize_node;
    // Step through the packets
    int ipkt, inode;
    for (ipkt=0; ipkt<npkt_node; ipkt++) {
        for (inode=0; inode<nnode; inode++) {
            char *pkt_p0 = nodebuf[inode] + ipkt*pktsize_bytes;
            char *pkt_p1 = pkt_p0 + pktsize_bytes/2;
            char *obase = out + (ipkt*nnode + inode)*pktsize_bytes;
            for (i=0; i<pktsize_bytes/16; i++) {
                // Group of 16 bytes at a time
                
#if 0 
                // This didn't work:
                obase[0] = pkt_p0[0];  
                obase[1] = pkt_p1[0];  
                obase[2] = pkt_p0[4];  
                obase[3] = pkt_p1[4];  
                obase[4] = pkt_p0[1];  
                obase[5] = pkt_p1[1];  
                obase[6] = pkt_p0[5];  
                obase[7] = pkt_p1[5];  
                obase[8]  = pkt_p0[2]; 
                obase[9]  = pkt_p1[2]; 
                obase[10] = pkt_p0[6]; 
                obase[11] = pkt_p1[6]; 
                obase[12] = pkt_p0[3]; 
                obase[13] = pkt_p1[3]; 
                obase[14] = pkt_p0[7]; 
                obase[15] = pkt_p1[7]; 
#endif

                // Try this one:
                obase[0] = pkt_p0[0];  
                obase[1] = pkt_p1[0];  
                obase[2] = pkt_p0[1];  
                obase[3] = pkt_p1[1];  
                obase[4] = pkt_p0[2];  
                obase[5] = pkt_p1[2];  
                obase[6] = pkt_p0[3];  
                obase[7] = pkt_p1[3];  
                obase[8]  = pkt_p0[4]; 
                obase[9]  = pkt_p1[4]; 
                obase[10] = pkt_p0[5]; 
                obase[11] = pkt_p1[5]; 
                obase[12] = pkt_p0[6]; 
                obase[13] = pkt_p1[6]; 
                obase[14] = pkt_p0[7]; 
                obase[15] = pkt_p1[7]; 

                obase += 16;
                pkt_p0 += 8;
                pkt_p1 += 8;
            }
        }
    }
}

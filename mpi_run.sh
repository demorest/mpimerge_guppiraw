# Note:  run as gpu!
mpirun -np 9 --hostfile /home/pulsar64/src/psrfits_utils/gpu_hostfile.txt /home/pulsar64/src/psrfits_utils/mpimerge_psrfits guppi_55317_Terzan5_0003

# Given the RHEL6 issues, we need to run on leibniz:
mpirun -np 9 --mca btl_tcp_if_include eth2 --hostfile /home/pulsar64/src/psrfits_utils/gpu_hostfile.txt /home/pulsar64/src/psrfits_utils/mpimerge_psrfits guppi_55317_Terzan5_0003



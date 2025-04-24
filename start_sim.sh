#!/bin/bash

SESSION_NAME="gz_session"

#!/bin/bash

gnome-terminal \
  --tab -- bash -c 'gz sim -v4 -r delivery_sim.sdf; exec bash' \
  --tab -- bash -c 'sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --map --console; exec bash'

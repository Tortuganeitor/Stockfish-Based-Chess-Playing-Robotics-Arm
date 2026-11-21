
import subprocess

print("THE PROGRAM STARTS RUNNING CORRECTLY")

#VARIBLES LEADER:
leader_port = "/dev/ttyACM1"
leader_name = "my_leader_arm"

#VARIBLES FOLLOWER:
follower_port = "/dev/ttyACM0"
follower_name = "my_follower_arm"

#CALIBRAR:
subprocess.run([
    "lerobot-calibrate",
    "--robot.type=so101_follower",
    f"--robot.port={leader_port}",
    f"--robot.id={leader_name}",
], check=True)

subprocess.run([
    "lerobot-calibrate",
    "--robot.type=so101_follower",
    f"--robot.port={follower_port}",
    f"--robot.id={follower_name}",
], check=True)

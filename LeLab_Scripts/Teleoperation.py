import subprocess

#VARIABLES LEADER:
leader_port = "/dev/ttyACM1"
leader_name = "my_leader_arm"

#VARIABLES FOLLOWER:
follower_port = "/dev/ttyACM0"
follower_name = "my_follower_arm"

subprocess.run([
    "lerobot-teleoperate",
    "--robot.type=so101_follower",
    f"--robot.port={follower_port}",
    f"--robot.id={follower_name}",
    "--robot.cameras={ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }",
    "--teleop.type=so101_leader",
    f"--teleop.port={leader_port}",
    f"--teleop.id={leader_name}",
    "--display_data=true",
], check=True)
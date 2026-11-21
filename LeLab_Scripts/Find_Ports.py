import subprocess

print("THE PROGRAM STARTS RUNNING CORRECTLY")

#DETECT AVAILABLE PORTS AND CAMERAS:
subprocess.run(["lerobot-find-port"], check=True)
#subprocess.run(["lerobot-find-cameras"], check=True)
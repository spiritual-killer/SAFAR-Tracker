# Create the shell script
nano install_dependencies.sh

# Add the following content to the file
#!/bin/bash

# Install system dependencies
apt-get update
apt-get install -y libgl1-mesa-glx

# Install Python dependencies
pip install -r requirements.txt

# Save and close the file

# Give execute permissions
chmod +x install_dependencies.sh

# Verify permissions
ls -l install_dependencies.sh

# Add the file to the staging area
git add install_dependencies.sh

# Commit the changes with a message
git commit -m "Add install_dependencies.sh script"

# Push the changes to the main branch on GitHub
git push origin main

#!/bin/bash
echo "Removing massspecgym"
pip uninstall massspecgym -y
echo "Updating massspecgym from GitHub..."
pip install --upgrade git+https://github.com/Jozefov/MassSpecGymMSn.git@main#egg=massspecgym
echo "massspecgym has been updated successfully."
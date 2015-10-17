#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
export AWS_SECRET_KEY=X8Z/K1EFbt71QwzrC/gFIYYYjtGTn2pwia5soZVo
export AWS_ACCESS_KEY=AKIAJCA4WFUJ52EC2IXA
cd /home/ubuntu/cheapgpu
curl https://s3.amazonaws.com/mniststartupscript/mnist.py > ./mnist.py
cat ./mnist.py
sudo -E python ./mnist.py
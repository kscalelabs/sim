echo "Starting download of CMU_fbx.zip"
aws s3 cp s3://kscale-public/expressive_humanoid/CMU_fbx.zip . &
echo "Starting download of fbx202032_fbxpythonbindings_linux.tar.gz"
aws s3 cp s3://kscale-public/expressive_humanoid/fbx202032_fbxpythonbindings_linux.tar.gz . &
echo "Starting download of fbx202032_fbxsdk_linux.tar.gz"
aws s3 cp s3://kscale-public/expressive_humanoid/fbx202032_fbxsdk_linux.tar.gz . &
echo "Starting download of sip-4.19.3.tar.gz"
aws s3 cp s3://kscale-public/expressive_humanoid/sip-4.19.3.tar.gz . &

wait
echo "All downloads completed"
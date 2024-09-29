PID=$(ps aux | grep 'run_exp.py' | grep -v grep | awk '{print $2}')
sleep 1

while kill -0 $PID 2>/dev/null; do
	echo "Process $PID is still running..."
	sleep 100  # 每10秒检查一次
done

echo "Process $PID has finished. Starting the next experiment."

    # 运行下一个实验
python3 run_exp.py decentralized
python3 run_exp.py distributed

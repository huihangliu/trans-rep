#!/bin/zsh
# Get the number of CPU cores and subtract 10
max_running_processes=$(( $(nproc) - 5 ))


# Function to check if there is enough free memory (5GB)
get_free_memory_percent() {
	local total_memory_mb=$(free -m | awk '/^Mem:/{print $2}')
	local used_memory_mb=$(free -m | awk '/^Mem:/{print $3}')
	local free_memory_percent=$((100 - (used_memory_mb * 100 / total_memory_mb)))

	echo $free_memory_percent # Output the free memory percentage

	if [ $free_memory_percent -ge 20 ]; then # e.g., 20% free memory required
		return 0 # True, enough memory available
	else
		return 1 # False, not enough memory
	fi
}

sequence_n=(200 400) # 20 30 50 100 200 400 600 800 1200 1600 2000
sequence_r=(5) # 1 3 5 7 9
sequence_p=(10 20) # 10 20 30 40
sequence_exp_id=(6) # 6 8 9 10


# Define script name and output directory as variables
SCRIPT_NAME="sim_main.py"
OUTPUT_DIR="./result/exp_2/"



# --------------------------------------------------------------------------------
for id in "${sequence_exp_id[@]}"; do
	for r in "${sequence_r[@]}"; do
		for n in "${sequence_n[@]}"; do
			for p in "${sequence_p[@]}"; do
				if [ $n -eq 200 ] && [ $p -eq 10 ]; then
					continue
				fi
				if [ $n -eq 400 ] && [ $p -eq 20 ]; then
					continue
				fi
				for i in {1..220}; do
					terminal_width=$(tput cols)
					while true; do
						current_running_processes=$(ps -eo s= | grep -c 'R')
						free_memory_percent=$(get_free_memory_percent)
						used_memory_percent=$((100 - free_memory_percent))
						printf "\e[1;34m%*s\e[0m\r" $terminal_width "[id=$id, n=$n, p=$p, r=$r, loop=$i; ${used_memory_percent}%Mem]"
						if [ $current_running_processes -lt $max_running_processes ] && [ $free_memory_percent -ge 20 ]; then
							nohup python "$SCRIPT_NAME" --n_source "$n" --p "$p" --r "$r" --seed "$i" --exp_id "$id" --record_dir "$OUTPUT_DIR" --verbose 0 >> "${OUTPUT_DIR}out.txt" 2>&1 &
							sleep_time=$((1)) # $(( n < 10 ? n : 10 ))
							sleep $sleep_time
							break
							# Exit the while loop after starting the Python process
						else
							sleep 5
						fi
					done
				done
			done
			rm "${OUTPUT_DIR}out.txt"
		done
	done
done

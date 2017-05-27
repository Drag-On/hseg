#! /bin/bash

server=atcremers50
directory=/work/moellerj/external

echo " => Updating project on $server..."
rsync -a --info=progress2 cmake exec include lib net src test CMakeLists.txt "$server:$directory"
ssh -o StrictHostkeyChecking=no -x "$server" bash << 'EOF'
	directory=/work/moellerj/external
	build_type=RelWithDebInfo
	build_threads=4
	run_app=hseg_train
	make_target=hseg_train

	if [[ ! -d "$directory/build" ]] ; then
		echo " => Build directory does not exist, creating new one."
		mkdir -p "$directory/build"
		cd "$directory/build"
		cmake -DCMAKE_BUILD_TYPE="$build_type" ..
	fi
	if [[ ! -d "$directory/out" ]] ; then
		mkdir "$directory/out"
	fi
	echo " => Building..."
    cd "$directory/build" ; make "$make_target" -j"$build_threads"
    if [[ $? -ne 0 ]] ; then
    	echo " => Make failed, try again."
    	cd "$directory/build" ; make "$make_target" -j"$build_threads"
    	if [[ $? -ne 0 ]] ; then
    		echo " => Make failed again :("
    		exit 1
    	fi
    fi
    echo " => Executing $run_app..."
    exec "$directory/build/$run_app"
EOF
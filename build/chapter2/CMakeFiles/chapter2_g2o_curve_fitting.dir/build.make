# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hange/Learn/BasicAlgorithmTest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hange/Learn/BasicAlgorithmTest/build

# Include any dependencies generated for this target.
include chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/depend.make

# Include the progress variables for this target.
include chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/progress.make

# Include the compile flags for this target's objects.
include chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/flags.make

chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.o: chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/flags.make
chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.o: ../chapter2/chapter2_g2o_curve_fitting.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/hange/Learn/BasicAlgorithmTest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.o"
	cd /home/hange/Learn/BasicAlgorithmTest/build/chapter2 && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.o -c /home/hange/Learn/BasicAlgorithmTest/chapter2/chapter2_g2o_curve_fitting.cpp

chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.i"
	cd /home/hange/Learn/BasicAlgorithmTest/build/chapter2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/hange/Learn/BasicAlgorithmTest/chapter2/chapter2_g2o_curve_fitting.cpp > CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.i

chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.s"
	cd /home/hange/Learn/BasicAlgorithmTest/build/chapter2 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/hange/Learn/BasicAlgorithmTest/chapter2/chapter2_g2o_curve_fitting.cpp -o CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.s

# Object files for target chapter2_g2o_curve_fitting
chapter2_g2o_curve_fitting_OBJECTS = \
"CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.o"

# External object files for target chapter2_g2o_curve_fitting
chapter2_g2o_curve_fitting_EXTERNAL_OBJECTS =

chapter2/chapter2_g2o_curve_fitting: chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/chapter2_g2o_curve_fitting.cpp.o
chapter2/chapter2_g2o_curve_fitting: chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/build.make
chapter2/chapter2_g2o_curve_fitting: chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/hange/Learn/BasicAlgorithmTest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable chapter2_g2o_curve_fitting"
	cd /home/hange/Learn/BasicAlgorithmTest/build/chapter2 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/chapter2_g2o_curve_fitting.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/build: chapter2/chapter2_g2o_curve_fitting

.PHONY : chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/build

chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/clean:
	cd /home/hange/Learn/BasicAlgorithmTest/build/chapter2 && $(CMAKE_COMMAND) -P CMakeFiles/chapter2_g2o_curve_fitting.dir/cmake_clean.cmake
.PHONY : chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/clean

chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/depend:
	cd /home/hange/Learn/BasicAlgorithmTest/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hange/Learn/BasicAlgorithmTest /home/hange/Learn/BasicAlgorithmTest/chapter2 /home/hange/Learn/BasicAlgorithmTest/build /home/hange/Learn/BasicAlgorithmTest/build/chapter2 /home/hange/Learn/BasicAlgorithmTest/build/chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : chapter2/CMakeFiles/chapter2_g2o_curve_fitting.dir/depend

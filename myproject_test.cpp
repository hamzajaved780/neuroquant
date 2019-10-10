//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "firmware/parameters.h"
#include "firmware/myproject.h"

#define Num_Invocation 10

int main(int argc, char **argv)
{
  //hls-fpga-machine-learning insert data
  hls_register inputdat input_1[Num_Invocation];
  hls_register outputdat output_1[Num_Invocation];

  for (int i = 0; i < Num_Invocation; ++i) {
    for(int j = 0; j < N_INPUT_1_1; ++j) {
      input_1[i].data[j] = 0;
    }
    //hls-fpga-machine-learning insert top-level-function
    ihc_hls_enqueue(&output_1[i], myproject, input_1[i]);
  }
  ihc_hls_component_run_all(myproject);

  //hls-fpga-machine-learning insert output
  for (int i = 0; i < Num_Invocation; ++i) {
    std::cout << "Input: ";
    for(int j = 0; j < N_INPUT_1_1; j++) {
      std::cout << input_1[i].data[j] << " ";
    }
    std::cout << ";";
    std::cout << "Output: ";
    for(int j = 0; j < N_LAYER_8; j++) {
      std::cout << output_1[i].data[j] << " ";
    }
    std::cout << ";";
    std::cout << std::endl;
  }

  return 0;
}

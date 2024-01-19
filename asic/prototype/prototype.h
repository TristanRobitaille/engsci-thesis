#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>

/*----- NAMESPACES -----*/
using namespace std;

/*----- DEFINES -----*/
#define CIM_WEIGHTS_STORAGE_SIZE_KB 2048
#define CIM_TEMP_STORAGE_SIZE_KB 512
#define CENTRALIZED_STORAGE_WEIGHTS_KB 2048

#define PATCH_LENGTH_NUM_SAMPLES 256

/*----- STRUCT -----*/
struct instruction {
    int op;
    int data;
};

/*----- ENUM -----*/
enum ops {

};

/*----- CLASSES -----*/
class CiM {
    private:
        uint16_t weights[CIM_WEIGHTS_STORAGE_SIZE_KB];
        float temp_storage[CIM_TEMP_STORAGE_SIZE_KB];

    public:
        int reset();
        int update();
};

class master_ctrl {
    private:
        int broadcast_inst();
        
    public:
        int reset();
        int load_signal();
        int load_weights();
};

/*----- GLOBALS -----*/
float centralized_storage[CENTRALIZED_STORAGE_WEIGHTS_KB];

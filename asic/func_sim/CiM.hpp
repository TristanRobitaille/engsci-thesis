#include <iostream>

/*----- NAMESPACE -----*/
using namespace std;

/*----- DEFINES -----*/
#define NUM_CIM 64
#define CIM_WEIGHTS_STORAGE_SIZE_KB 2048
#define CIM_TEMP_STORAGE_SIZE_KB 512

/*----- CLASS -----*/
class CiM {
    private:
        uint16_t weights[CIM_WEIGHTS_STORAGE_SIZE_KB];
        float temp_storage[CIM_TEMP_STORAGE_SIZE_KB];

    public:
        CiM();
        int reset();
        int update();
};
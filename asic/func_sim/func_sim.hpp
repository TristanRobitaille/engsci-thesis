#include <iostream>
#include "CiM.hpp"
#include "Master_Ctrl.hpp"

/*----- NAMESPACE -----*/
using namespace std;

/*----- DEFINES -----*/
#define CENTRALIZED_STORAGE_WEIGHTS_KB 2048
#define PATCH_LENGTH_NUM_SAMPLES 256

/*----- STRUCT -----*/
struct instruction {
    int op;
    int data;
};

/*----- ENUM -----*/
enum ops {};

/*----- GLOBALS -----*/
float centralized_storage[CENTRALIZED_STORAGE_WEIGHTS_KB];

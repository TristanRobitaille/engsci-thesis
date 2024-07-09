
BOOSTSRC := lib/boost_1_84_0
N_STO_INT_RES ?= 20
N_STO_PARAMS ?= 20

default: build

.PHONY: clean
clean:
	rm -rf build

.PHONY: buildBoost
buildBoost:
	cd ${BOOSTSRC}; ./bootstrap.sh --prefix=${BOOSTINSTALLED}
	cd ${BOOSTSRC}; ./b2 install

.PHONY: configure
configure:
	@cmake -S . -B build -DBOOSTINSTALLED=$(CURDIR)/../boost -DN_STO_INT_RES=${N_STO_INT_RES} -DN_STO_PARAMS=${N_STO_PARAMS}

.PHONY: configureNoBoost
configureNoBoost:
	@cmake -S . -B build -DN_STO_INT_RES=${N_STO_INT_RES} -DN_STO_PARAMS=${N_STO_PARAMS}

.PHONY: build
build:
	@cmake --build build

.PHONY: reconfigure
reconfigure: clean configure


BOOSTSRC := lib/boost_1_84_0
N_STO ?= 20

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
	@cmake -S . -B build -DBOOSTINSTALLED=$(CURDIR)/../boost -DN_STO=${N_STO}

.PHONY: configureNoBoost
configureNoBoost:
	@cmake -S . -B build -DN_STO=${N_STO}

.PHONY: build
build:
	@cmake --build build

.PHONY: reconfigure
reconfigure: clean configure

SHELL = /bin/bash
TARGET = harness

FC = gfortran

# NetCDF paths — use nf-config if available, otherwise system defaults
NF_FFLAGS  := $(shell nf-config --fflags  2>/dev/null)
NF_FLIBS   := $(shell nf-config --flibs   2>/dev/null)
ifeq ($(NF_FFLAGS),)
  NF_FFLAGS = -I/usr/include
endif
ifeq ($(NF_FLIBS),)
  NF_FLIBS = -lnetcdff -lnetcdf
endif

# Must match upstream SVMC flags — especially -freal-4-real-8
FFLAGS = $(NF_FFLAGS) -O2 -g -cpp -m64 -freal-4-real-8 -DUSE_NCF -fcheck=all
LDFLAGS = $(FFLAGS) $(NF_FLIBS)

SVMC_SRC = ../../vendor/SVMC/src

# Module objects (order matters for dependencies)
MODOBJS = \
	readctrl_mod.o \
	readvegpara_mod.o \
	readsoilpara_mod.o \
	yasso.o \
	yassofortran20.o \
	phydro_mod.o \
	water_mod.o \
	blas.o \
	linpack.o \
	timer.o \
	lbfgsb.o

OBJECTS = harness.o

$(TARGET): $(MODOBJS) $(OBJECTS)
	$(FC) $(MODOBJS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

harness.o: harness.f90 $(MODOBJS)
	$(FC) -c $(FFLAGS) $<

# Compile SVMC sources from vendor directory
readctrl_mod.o: $(SVMC_SRC)/readctrl_mod.f90
	$(FC) -c $(FFLAGS) $<

readvegpara_mod.o: $(SVMC_SRC)/readvegpara_mod.f90 readctrl_mod.o
	$(FC) -c $(FFLAGS) $<

readsoilpara_mod.o: $(SVMC_SRC)/readsoilpara_mod.f90
	$(FC) -c $(FFLAGS) $<

phydro_mod.o: $(SVMC_SRC)/phydro_mod.f90 readctrl_mod.o readvegpara_mod.o
	$(FC) -c $(FFLAGS) $<

water_mod.o: $(SVMC_SRC)/water_mod.f90 readctrl_mod.o readvegpara_mod.o readsoilpara_mod.o
	$(FC) -c $(FFLAGS) $<

blas.o: $(SVMC_SRC)/blas.f
	$(FC) -c $(FFLAGS) $<

linpack.o: $(SVMC_SRC)/linpack.f
	$(FC) -c $(FFLAGS) $<

lbfgsb.o: $(SVMC_SRC)/lbfgsb.f
	$(FC) -c $(FFLAGS) $<

timer.o: $(SVMC_SRC)/timer.f
	$(FC) -c $(FFLAGS) $<

yasso.o: $(SVMC_SRC)/yasso.f90
	$(FC) -c $(FFLAGS) $<

yassofortran20.o: $(SVMC_SRC)/yassofortran20.f90
	$(FC) -c $(FFLAGS) $<

clean:
	rm -f *.o *.mod $(TARGET)

.PHONY: clean

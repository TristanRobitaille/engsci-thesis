clear &&
rm -r __pycache__ &&
rm -r sim_build &&
rm dump.vcd &&
rm results.xml &&
make &&
gtkwave dump.vcd
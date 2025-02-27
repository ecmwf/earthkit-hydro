import earthkit.hydro as ekh


def read_network(reader, map_name):
    if "d8_ldd" in reader:
        network = ekh.from_d8(map_name)
    elif "cama_downxy" in reader:
        network = ekh.from_cama_downxy(*map_name)
    elif "cama_nextxy" in reader:
        network = ekh.from_cama_nextxy(*map_name)
    return network

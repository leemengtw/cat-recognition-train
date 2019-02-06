import os


def main(path):
    os.system("benchmark_model --graph=%s --show_flops --input_layer=x --input_layer_type=float --input_layer_shape=1,224,224,3 --output_layer=net_graph/Reshape_1" % path)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])

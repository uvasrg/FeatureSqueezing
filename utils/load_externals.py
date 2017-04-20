import sys, os

external_libs = {'Cleverhans v1.0.0': "externals/cleverhans",
                 'Tensorflow-Model-Resnet': "externals/tensorflow-models",
                 }

project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

for lib_name, lib_path in external_libs.iteritems():
    lib_path = os.path.join(project_path, lib_path)
    if os.listdir(lib_path) == []:
        cmd = "git submodule update --init --recursive"
        print("Fetching external libraries...")
        os.system(cmd)

    if lib_name == 'Tensorflow-Model-Resnet':
        lib_token_fpath = os.path.join(lib_path, 'resnet', '__init__.py')
        if not os.path.isfile(lib_token_fpath):
            open(lib_token_fpath, 'a').close()
    
    sys.path.append(lib_path)
    print("Located %s" % lib_name)

# print (sys.path)

import os

def get_names(path_to_names):
    names = {}
    folders = os.listdir(path_to_names)
    i = 0
    for folder in folders:
        names[i] = str(folder)
        i += 1

    return names


if __name__=="__main__":

    a = r"C:\Users\anura\Desktop\hcthn_prjct\simpsons\simpsons_dataset"
    names = get_names(a)
    print(names[0])

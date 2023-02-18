import os


def get_filename_from_path(path, with_type=True):
    _, file_name = os.path.split(path)
    if with_type:
        return file_name
    else:
        return os.path.splitext(file_name)[0]


def get_file_list(dir, types, is_full_path):
    files = []
    for i in types:
        files.append([])
    for i in os.listdir(dir):
        temp = os.path.join(dir, i)
        if os.path.isdir(temp):
            sub_list = get_file_list(temp, types, is_full_path)
            for k, _ in enumerate(types):
                files[k] += sub_list[k]
        else:
            suffix = os.path.splitext(i)[-1][1:]
            if suffix in types:
                if is_full_path:
                    files[types.index(suffix)].append(temp)
                else:
                    files[types.index(suffix)].append(i)
    return files


if __name__ == '__main__':
    files = get_file_list('/home/alliance/Files/Dataset/DJI_ROCO', ['xml', 'jpg'], True)

    print(files)

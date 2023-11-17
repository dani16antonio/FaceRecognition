def extract_faces(dataset_path):
    image_extention = ("jpg", "png", "jpeg")
    get_label = lambda path_file: path_file.split(os.sep)[-2]
    X, y = [], []
    
    for file_name in glob(os.sep.join([dataset_path, "**"]), recursive=True):
        if file_name.split(".")[-1].lower() not in image_extention:
            continue
        print(file_name)
        X.append(extract_face(file_name))
        y.append(get_label(file_name))
    
    return np.array(X), np.array(y)                 
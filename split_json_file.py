import json

if __name__ == "__main__":
    # in_file_path = 'JerichoWorld-main/data/test.json'  # Change me!
    #
    # name_list = []
    # with open(in_file_path, 'r') as in_json_file:
    #
    #     # Read the file and convert it to a dictionary
    #     json_obj_list = json.load(in_json_file)
    #
    #     for json_obj in json_obj_list:
    #         name = (json_obj[0]["rom"]).upper()
    #         filename = "JerichoWorld-main/data/separate/" + name + '.json'
    #         name_list.append(name)
    #
    #         with open(filename, 'w') as out_json_file:
    #             game = json_obj
    #             list_with_one_game = [json_obj]
    #             json.dump(list_with_one_game, out_json_file, indent=4)
    #
    # with open("JerichoWorld-main/data/all_names.txt", 'a') as out_json_file:
    #     out_json_file.write("\n".join(str(item) for item in name_list))

    in_file_path_train = 'JerichoWorld-main/data/train.json'
    in_file_path_test = 'JerichoWorld-main/data/test.json'

    name_list = []
    with open(in_file_path_train, 'r') as in_json_file_train:
        with open(in_file_path_test, 'r') as in_json_file_test:

            json_obj_list_train = json.load(in_json_file_train)
            json_obj_list_test = json.load(in_json_file_test)

            json_obj_list_train.extend(json_obj_list_test)
            joined_list = json_obj_list_train
#            for json_obj in json_obj_list_test:
#                json_obj_list_train.extend(json_obj)

            with open("JerichoWorld-main/data/separate/ALL.json", 'w') as out_json_file:
#                 list_with_one_game = [json_obj_list_train]
                json.dump(joined_list, out_json_file, indent=4)

            with open("JerichoWorld-main/data/no_samples", 'w') as out_json_file:
                for json_obj in joined_list:
                    name = (json_obj[0]["rom"]).upper()
                    out_json_file.write("\n" + str(name) + ": " + str(len(json_obj)))

            #         filename = "JerichoWorld-main/data/separate/" + name + '.json'
            #         name_list.append(name)

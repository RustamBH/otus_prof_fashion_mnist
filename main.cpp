#include <iostream>
#include <fstream>

#include "logreg_classifier.h"
#include "helpers.h"

using kdd99::LogregClassifier;

int main(int argc, char *argv[])
{
    if (argc != 3) {
        std::cerr << "Input: fashion_mnist <input_csv_file> <logreg_coef_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::vector<LogregClassifier> classifiers;
    LogregClassifier::coef_t coefs;

	// читеам коэффициенты
    std::ifstream coef_input(argv[2]);
    while (true)
    {
        if (!read_coefs(coef_input, coefs))
            break;

        classifiers.emplace_back(LogregClassifier(coefs));
    }
    coef_input.close();
	
	// читеам данные
    std::ifstream data_input(argv[1]);

    LogregClassifier::features_t features;
    int target_class;
    int total_cnt = 0;
    int right_ans_cnt = 0;

    while (true) {
        if (!read_features(data_input, features, target_class))
            break;

        total_cnt++;

        float max_result = -1;
        int max_res_class = 0;

        for (size_t i = 0; i < classifiers.size(); i++)
        {
            auto result = classifiers[i].predict_proba(features);
            if (result > max_result)
            {
                max_result = result;
                max_res_class = i;
            }
        }

        if (max_res_class == target_class)
            right_ans_cnt++;
    }

    float accuracy = 0;
    if (total_cnt > 0)
    {
        accuracy = float(right_ans_cnt) / total_cnt;
    }

    std::cout << accuracy << std::endl;

    return 0;
}

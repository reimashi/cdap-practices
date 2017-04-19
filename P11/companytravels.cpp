#include <iostream>
#include <cstring>
#include <vector>
#include <sstream>

const char KEY_DELIMITER = '|';
const char DELIMITER = '\t';

std::string trim(const std::string& str)
{
    static std::string whitespaces (" \t\f\v\n\r");
    size_t first = str.find_first_not_of(whitespaces);
    if (std::string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(whitespaces);
    return str.substr(first, (last - first + 1));
}

std::vector<std::string> split(const std::string& str, char chr) {
    std::vector<std::string> vect;
    std::stringstream ss(str);
    std::string elem;

    while(std::getline(ss, elem, chr)) {
        vect.push_back(elem);
    }

    return vect;
}

int mapper() {
    char buffer[256];

    while (std::cin.getline(buffer, sizeof(buffer))) {
        std::string line = trim(std::string(buffer));
        std::vector<std::string> lineSplit = split(line, ',');

        if (lineSplit.size() != 2) {
            std::cout << "Format error. Input: " << line << std::endl;
        }
        else {
            std::string key = trim(lineSplit[0]);

            try {
                int32_t value = std::stoi(lineSplit[1]);
                // Si se puede convertir a entero, es un join2_num*
                std::cout << key << KEY_DELIMITER << "Iberia" << DELIMITER << value << std::endl;
                std::cout << key << KEY_DELIMITER << "Carrefour" << DELIMITER << value << std::endl;
            }
            catch (std::exception e) {
                // Si falla la conversion a entero, es un join2_dest*
                if (lineSplit[1].compare("Iberia") == 0 || lineSplit[1].compare("Carrefour") == 0)
                    std::cout << key << KEY_DELIMITER << lineSplit[1] << DELIMITER << -1 << std::endl;
            }
        }
    }
}

int reducer() {
    char buffer[256];

    std::string lastKey = "";
    int32_t lastKeyValue = 0;

    while (std::cin.getline(buffer, sizeof(buffer))) {
        std::string line = trim(std::string(buffer));
        std::vector<std::string> lineSplit = split(line, DELIMITER);

        if (lineSplit.size() != 2) {
            std::cout << "Format error. Input: " << line << std::endl;
        }
        else {
            try {
                std::string key = trim(lineSplit[0]);
                int32_t value = std::stoi(lineSplit[1]);

                if (key.compare(lastKey) == 0) {
                    lastKeyValue += value;
                }
                else {
                    if (lastKey.compare("") != 0 && lastKeyValue > 0) std::cout << lastKey << DELIMITER << lastKeyValue << std::endl;

                    // Si hay un registro de destino marcado por -1, se reducen. Si no se descartan.
                    if (value == -1) {
                        lastKey = key;
                        lastKeyValue = value;
                    }
                    else {
                        lastKey = "";
                    }
                }
            }
            catch(std::exception e) {
                std::cout << "Format error. The second element must be a number. Input: " << lineSplit[1] << std::endl;
            }
        }
    }

    if (std::cin.eof() && lastKey.compare("") != 0 && lastKeyValue > 0) {
        std::cout << lastKey << DELIMITER << lastKeyValue << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "--map") == 0) mapper();
    else if (argc == 2 && strcmp(argv[1], "--reduce") == 0) reducer();
    else std::cout << "Select the --map or --reduce option" << std::endl;
    return 0;
}
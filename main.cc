#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#define IMG_FILE          "mnist/SOM_MNIST_data.txt"
#define IMG_W              28
#define IMG_H              28
#define VEC_SIZE           (IMG_W * IMG_H)
#define IMG_MAX_VAL        255
#define NUM_IMG            5000

using namespace std;

typedef double Pixel;
typedef Pixel Image[VEC_SIZE];    /* Image is a vector. */

typedef struct S_Data {
    Image* img;
    unsigned int size;
} Data;

typedef struct S_Coordinate {
    unsigned int x;
    unsigned int y;
} Coordinate;

// Read bytes from file and cast to type T
template <typename T>
T read_to(ifstream &fp, bool msb_first=true);
template <typename T>
T read_to(ifstream &fp, bool msb_first) {
    size_t size = sizeof(T);
    char buf[size + 1];
    fp.get(buf, size + 1);
    T val;
    if (msb_first)
       for (int i = 0; i < size / 2; i++)
           swap(buf[i], buf[size-1-i]);
    memcpy(&val, buf, sizeof(T));
    return val;
}

Data read_data_from_file(string img_file) {
    Data data = {NULL, 0};
    ifstream fp(img_file.c_str());
    data.size = NUM_IMG;
    data.img = new Image[NUM_IMG];
    
    for (auto i = 0; i < VEC_SIZE; i++) {
        for (auto j = 0; j < NUM_IMG; j++)
            fp >> data.img[j][i];
        cout << "\rDim: " << i;
    }
    cout << endl;
    fp.close();
    
    cout << "Successfully read " << data.size << " images and labels." << endl;
    return data;
}

void normalize_image(Image &img) {
    for (auto i = 0; i < VEC_SIZE; i++)
        img[i] /= (double)IMG_MAX_VAL;
}

void denormalize_image(Image &img) {
    for (auto i = 0; i < VEC_SIZE; i++)
        img[i] *= (double)IMG_MAX_VAL;
}

Image &copy_image(Image &result, const Image &src) {
    for (auto i = 0; i < VEC_SIZE; i++)
        result[i] = src[i];
    return result;
}

void free_data(Data &data) {
    delete [] data.img;
    data.img = NULL;
    data.size = 0;
}

void show_img(const Image &img, Pixel threshold=0.5) {
    for (auto i = 0; i < VEC_SIZE; i++)
        cout << " #"[img[i] > threshold] 
             << ((i + 1) % IMG_W == 0 ? "\n" : "");
}

void save_bmp(const Image &img, const char *name=NULL) {
    string file_name = "";
    file_name += name ? name : "output";
    file_name += ".bmp";
    ofstream fp(file_name.c_str(), std::ios::binary);
    unsigned int offset = 0x36,
                 img_size = VEC_SIZE * 3,
                 file_size = img_size + offset,
                 width = IMG_W,
                 height = IMG_H,
                 empty = 0,
                 header_size = 40,
                 color_planes = 1,
                 num_per_pixel = 24;
    // BMP Header
    fp.write("BM", 2);
    fp.write((char *)&file_size, 4);
    fp.write((char *)&empty, 4);
    fp.write((char *)&offset, 4);
    // DIB
    fp.write((char *)&header_size, 4);
    fp.write((char *)&width, 4);
    fp.write((char *)&height, 4);
    fp.write((char *)&color_planes, 2);
    fp.write((char *)&num_per_pixel, 2);
    fp.write((char *)&empty, 4);
    fp.write((char *)&img_size, 4);
    fp.write((char *)&empty, 4);
    fp.write((char *)&empty, 4);
    fp.write((char *)&empty, 4);
    fp.write((char *)&empty, 4);
    // Image data
    for (int i = height-1; i >= 0; i--)
        for (int j = 0; j < width; j++) {
            double pixel = img[i * width + j];
            unsigned char pixel_char = pixel < 0 ? 0 : (
                          pixel > IMG_MAX_VAL ? IMG_MAX_VAL : pixel);
            fp.write((char *)&pixel_char, 1);
            fp.write((char *)&pixel_char, 1);
            fp.write((char *)&pixel_char, 1);
        }
    fp.close();
}

/* vector util */
double dot(const Image &a, const Image &b) {
    double sum = 0;
    for (auto i = 0; i < VEC_SIZE; i++)
        sum += a[i] * b[i];
    return sum;
}

Image &transpose_inplace(Image &a) {
    for (auto i = 0; i < IMG_H; i++)
        for (auto j = i; j < IMG_W; j++)
            swap(a[i * IMG_W + j], a[j * IMG_H + i]);
    return a;
}

Image &sub(Image &result, const Image &a, const Image &b) {
    for (auto i = 0; i < VEC_SIZE; i++)
        result[i] = a[i] - b[i];
    return result;
}

double norm(const Image &a) {
    double sum = 0;
    for (auto i = 0; i < VEC_SIZE; i++)
        sum += pow(a[i], 2);
    return sqrt(sum);
}

double distance(const Image &a, const Image &b) {
    // Euclidean distance
    Image result;
    sub(result, a, b);
    return norm(result);
}

double gaussian(double val, double sigma) {
    return exp(-val/(2 * sigma * sigma));
}

double random(double low, double high) {
    return double(rand()) / (RAND_MAX/(high - low)) + low;
}

double neighborhood(unsigned int bmu_x, unsigned int bmu_y, 
               unsigned int x, unsigned int y, double sigma) {
    return exp(-(pow(bmu_x - x, 2) + pow(bmu_y - y, 2)) / (2 * pow(sigma, 2)));
}

class SOM {
    public:
        SOM(Data &data, unsigned int grid_w, unsigned int grid_h, 
                 unsigned int total_step, double init_learning_rate) {
            weight = new Image *[grid_h];
            for (auto i = 0; i < grid_h; i++)
                weight[i] = new Image[grid_w];
                
            this->grid_w = grid_w;
            this->grid_h = grid_h;
            this->step = 0;
            this->data = data;
            this->total_step = total_step;
            this->learning_rate = this->init_learning_rate = init_learning_rate;
            this->sigma = this->init_sigma = sqrt(0.5*grid_h*grid_h + 0.5*grid_w*grid_w);
            //this->lambda = lambda;
            
            // init: Random pixels to weight
            for (auto i = 0; i < grid_h; i++)
                for (auto j = 0; j < grid_w; j++)
                    for (auto k = 0; k < VEC_SIZE; k++)
                        weight[i][j][k] = random(0, 1);
        }
        
        ~SOM() {
            for (auto i = 0; i < grid_h; i++)
                delete [] weight[i];
            delete [] weight;
        }
        
        void train() {
            unsigned int indices[NUM_IMG];
            for (int i = 0; i < NUM_IMG; i++)
                indices[i] = i;
                
            // Create old weight to store a copy of weight from last step
            Image **old_weight = new Image *[grid_h];
            for (auto i = 0; i < grid_h; i++)
                old_weight[i] = new Image[grid_w];
            
            for (int t = 0; t < total_step; t++) {
                // shuffle indices
                for (int i = NUM_IMG - 1; i > 0; i--)
                    swap(indices[i], indices[rand() % i]);
                    
                // Backup old weight
                for (auto x = 0; x < grid_h; x++)
                    for (auto y = 0; y < grid_w; y++)
                        copy_image(old_weight[x][y], weight[x][y]);
                    
                for (int idx_i = 0; idx_i < NUM_IMG; idx_i++) {
                    Image &img = data.img[indices[idx_i]];
                    // Train one image below:
                    Coordinate bmu_xy = activate(img);
                    
                    for (auto x = 0; x < grid_h; x++)
                        for (auto y = 0; y < grid_w; y++) {
                            Image &w = weight[x][y];
                            Image result;
                            sub(result, img, w);
                            double factor = learning_rate * neighborhood(bmu_xy.x, bmu_xy.y, x, y, sigma);
                            for (auto i = 0; i < VEC_SIZE; i++)
                                w[i] += factor * result[i];
                        }
                    // End of train one image
                }
                // Calculate difference of weight
                double diff_square = 0;
                for (auto x = 0; x < grid_h; x++)
                    for (auto y = 0; y < grid_w; y++)
                        for (auto k = 0; k < VEC_SIZE; k++)
                            diff_square += pow(weight[x][y][k] - old_weight[x][y][k], 2);
                diff_square /= (grid_w * grid_h * VEC_SIZE);
                double diff = sqrt(diff_square);
                
                cout << " Training (" << t+1 << "/" << total_step << ")  diff = " << diff << endl;
                // Update learning_rate and sigma
                learning_rate = learning_rate_func();
                sigma = sigma_func();
            }
            // Free old_weight
            for (auto i = 0; i < grid_h; i++)
                delete [] old_weight[i];
            delete [] old_weight;
        }
        
        Coordinate activate(const Image &img) {
            unsigned int bmu_x, bmu_y;
            double min_distance = DBL_MAX;
            // Calculate activated coor (BMU coordinate)
            for (auto x = 0; x < grid_h; x++)
                for (auto y = 0; y < grid_w; y++) {
                    double d = distance(weight[x][y], img);
                    if (d < min_distance) {
                        min_distance = d;
                        bmu_x = x;
                        bmu_y = y;
                    }
                }
            return {bmu_x, bmu_y};
        }
        
        double learning_rate_func() {
            return init_learning_rate / (1 + step/(total_step/2));
        }
        
        double sigma_func() {
            return init_sigma / (1 + step/(total_step/2));
        }
        
    private:
        Data data;
        unsigned int total_step;
        unsigned int step;
        unsigned int grid_w;
        unsigned int grid_h;
        Image **weight;
        double init_learning_rate;
        double init_sigma;
        double learning_rate;
        double sigma;
        // weight
};

void test() {
    unsigned int indices[6];
    for (int i = 0; i < 6; i++)
        indices[i] = i;
    // shuffle
    for (int i = 6 - 1; i > 0; i--)
        swap(indices[i], indices[rand() % i]);
    for (int i = 0; i < 6; i++)
        cout << indices[i] << " ";
    cout << endl;
}

int main() {
    srand ((unsigned int)time(0));
    Data data = read_data_from_file(IMG_FILE);
    for (auto i = 0; i < NUM_IMG; i++)
        transpose_inplace(data.img[i]);
    
    SOM som(data, 10, 10, 200, 0.5);
    som.train();
    
    ofstream output("activate_result.csv");
    output << "data_idx,bmu_x,bmu_y" << endl;
    for (auto i = 0; i < NUM_IMG; i++) {
        Coordinate bmu = som.activate(data.img[i]);
        output << i << "," << bmu.x << "," << bmu.y << endl;
    }
    output.close();
    
    free_data(data);
    
    return 0;
}


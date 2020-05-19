#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cmath>
#include <ctime>

#define TST_LBL_FILE      "mnist/t10k-labels.idx1-ubyte"
#define TST_IMG_FILE      "mnist/t10k-images.idx3-ubyte"
#define TRN_LBL_FILE      "mnist/train-labels.idx1-ubyte"
#define TRN_IMG_FILE      "mnist/train-images.idx3-ubyte"
#define LBL_MAGIC          2049
#define IMG_MAGIC          2051
#define IMG_W              28
#define IMG_H              28
#define VEC_SIZE           (IMG_W * IMG_H)
#define IMG_MAX_VAL        255

using namespace std;

typedef unsigned char Label;
typedef double Pixel;
typedef Pixel Image[VEC_SIZE];    /* Image is a vector. */

typedef struct S_Data{
    Label* lbl;
    Image* img;
    unsigned int size;
} Data;



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

Data read_data_from_file(string lbl_file, string img_file) {
    Data data = {NULL, NULL, 0};
    // Read labels, check magic and store n_items
    ifstream f_lbl(lbl_file.c_str(), std::ios::binary);
    unsigned int lbl_magic = read_to<unsigned int>(f_lbl);
    unsigned int n_items = read_to<unsigned int>(f_lbl);
    if (lbl_magic != LBL_MAGIC)
       throw "Magic number error in label file.";
       
    data.size = n_items;
    data.lbl = new Label[n_items];
    
    for (auto i = 0; i < n_items; i++)
        data.lbl[i] = f_lbl.get();
    f_lbl.close();
    
    // Read images, check magic/n_items/img_w/img_h
    ifstream f_img(img_file.c_str(), std::ios::binary);
    unsigned int img_magic = read_to<unsigned int>(f_img);
    n_items = read_to<unsigned int>(f_img);
    unsigned int img_w = read_to<unsigned int>(f_img);
    unsigned int img_h = read_to<unsigned int>(f_img);
    if (img_magic != IMG_MAGIC)
       throw "Magic number error in image file.";
    if (n_items != data.size)
       throw "Size of image and label doesn't match.";
    if (img_w != IMG_W || img_h != IMG_H)
       throw "Invalid image width or height.";
    
    data.img = new Image[n_items];
    
    for (auto i = 0; i < n_items; i++)
        for (auto j = 0; j < VEC_SIZE; j++)
            data.img[i][j] = (double)f_img.get();
        
    f_img.close();
    
    cout << "Successfully read " << data.size << " images and labels." << endl;
    return data;
}

void normalize_image(Image &img) {
    for (auto i = 0; i < VEC_SIZE; i++)
        i /= (double)IMG_MAX_VAL;
}

void denormalize_image(Image &img) {
    for (auto i = 0; i < VEC_SIZE; i++)
        i *= (double)IMG_MAX_VAL;
}

void free_data(Data &data) {
    delete [] data.lbl;
    delete [] data.img;
    data.lbl = NULL;
    data.img = NULL;
    data.size = 0;
}

void show_img(const Data &data, const unsigned int idx, Pixel threshold=127.0) {
    for (auto i = 0; i < VEC_SIZE; i++)
        cout << " #"[data.img[idx][i] > threshold] 
             << ((i + 1) % IMG_W == 0 ? "\n" : "");
    cout << "     (label = " << (int)data.lbl[idx] << ")" << endl;
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
            this->init_learning_rate = init_learning_rate;
            this->init_sigma = sqrt(0.5*grid_h*grid_h + 0.5*grid_w*grid_w);
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
        // weight
};

int main() {
    srand ((unsigned int)time(0));
    Data data = read_data_from_file(TST_LBL_FILE, TST_IMG_FILE);
    cout << norm(data.img[0]);
    SOM som(data, 2, 2, 200, 0.1);
    save_bmp(data.img[1]);
    free_data(data);
    
    return 0;
}


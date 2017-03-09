//
// Created by jan on 18.08.16.
//

#include <boost/unordered_map.hpp>
#include <typedefs.h>
#include <png.h>
#include <iostream>
#include "helper/image_helper.h"

namespace helper
{
    namespace image
    {
        ColorMap generateColorMapVOC(size_t n)
        {
            enum Bits
            {
                FirstBit = 1 << 0,
                SecondBit = 1 << 1,
                ThirdBit = 1 << 2,
            };
            ColorMap cmap(n);
            for (size_t i = 0; i < n; ++i)
            {
                size_t id = i;
                unsigned char r = 0, g = 0, b = 0;
                for (int j = 0; j <= 7; ++j)
                {
                    // Note: This is switched compared to the pascal voc example code because opencv has BGR instead of RGB
                    b = b | static_cast<unsigned char>(((id & FirstBit) >> 0) << (7 - j));
                    g = g | static_cast<unsigned char>(((id & SecondBit) >> 1) << (7 - j));
                    r = r | static_cast<unsigned char>(((id & ThirdBit) >> 2) << (7 - j));
                    id = id >> 3;
                }
                cmap[i][0] = r;
                cmap[i][1] = g;
                cmap[i][2] = b;
            }
            return cmap;
        }

        ColorMap generateColorMapCityscapes()
        {
            ColorMap cmap = {
                    {128, 64,128}, // 0 Road
                    {244, 35,232}, // 1 Sidewalk
                    { 70, 70, 70}, // 2 Building
                    {102,102,156}, // 3 Wall
                    {190,153,153}, // 4 Fence
                    {153,153,153}, // 5 Pole
                    {250,170, 30}, // 6 Traffic light
                    {220,220,  0}, // 7 Traffic sign
                    {107,142, 35}, // 8 Vegetation
                    {152,251,152}, // 9 Terrain
                    { 70,130,180}, // 10 Sky
                    {220, 20, 60}, // 11 Person
                    {255,  0,  0}, // 12 Rider
                    {  0,  0,142}, // 13 Car
                    {  0,  0, 70}, // 14 Truck
                    {  0, 60,100}, // 15 Bus
                    {  0, 80,100}, // 16 Train
                    {  0,  0,230}, // 17 Motorcycle
                    {119, 11, 32}, // 18 Bicycle
                    {111, 74,  0}, // 19 dynamic (ignored)
                    { 81,  0, 81}, // 20 Ground (ignored)
                    {250,170,160}, // 21 Parking (ignored)
                    {230,150,140}, // 22 Rail Track (ignored)
                    {180,165,180}, // 23 Guard Rail (ignored)
                    {150,100,100}, // 24 Bridge (ignored)
                    {150,120, 90}, // 25 Tunnel (ignored)
                    {  0,  0, 90}, // 26 Caravan (ignored)
                    {  0,  0,110}, // 27 Trailer (ignored)
                    {  0,  0,142}, // 28 Licence Plate (ignored)
                    {  0,  0,  0}, // 29 unlabeled (ignored)
            };
            return cmap;
        }

        ColorMap generateColorMap(size_t n)
        {
            static constexpr std::array<unsigned char, 3> colors[] = {{255,0,0}, {102,92,51}, {0,255,238}, {80,45,179}, {242,0,0}, {64,57,32}, {0,179,167}, {46,25,102}, {204,0,0}, {217,206,163}, {0,140,131}, {34,19,77}, {166,0,0}, {153,145,115}, {0,102,95}, {137,108,217}, {127,0,0}, {115,109,86}, {0,77,71}, {146,134,179}, {89,0,0}, {89,85,67}, {0,51,48}, {83,77,102}, {64,0,0}, {255,238,0}, {54,217,206}, {102,0,255}, {51,0,0}, {191,179,0}, {128,255,246}, {77,0,191}, {255,64,64}, {127,119,0}, {96,191,185}, {70,32,128}, {191,48,48}, {102,95,0}, {70,140,136}, {28,13,51}, {140,35,35}, {64,60,0}, {191,255,251}, {179,128,255}, {102,26,26}, {229,218,57}, {86,115,113}, {125,89,179}, {76,19,19}, {166,157,41}, {0,238,255}, {80,57,115}, {255,128,128}, {255,247,128}, {0,190,204}, {184,163,217}, {204,102,102}, {217,210,108}, {0,155,166}, {119,105,140}, {153,77,77}, {115,111,57}, {0,119,128}, {116,0,217}, {115,57,57}, {76,74,38}, {0,71,77}, {75,0,140}, {89,45,45}, {255,251,191}, {26,97,102}, {61,0,115}, {64,32,32}, {51,50,38}, {128,247,255}, {48,0,89}, {255,191,191}, {238,255,0}, {89,173,179}, {34,0,64}, {204,153,153}, {143,153,0}, {32,62,64}, {149,57,230}, {153,115,115}, {48,51,0}, {163,213,217}, {124,48,191}, {102,77,77}, {218,230,57}, {115,150,153}, {99,38,153}, {77,57,57}, {182,191,48}, {57,75,77}, {108,70,140}, {51,38,38}, {121,128,32}, {0,204,255}, {68,45,89}, {102,14,0}, {85,89,22}, {0,153,191}, {39,26,51}, {255,89,64}, {173,179,89}, {0,122,153}, {225,191,255}, {204,71,51}, {136,140,70}, {0,92,115}, {56,48,64}, {166,58,41}, {176,179,134}, {0,71,89}, {170,0,255}, {115,40,29}, {204,255,0}, {0,51,64}, {111,0,166}, {255,145,128}, {163,204,0}, {54,184,217}, {191,64,255}, {204,116,102}, {112,140,0}, {128,230,255}, {77,26,102}, {166,94,83}, {82,102,0}, {96,172,191}, {48,16,64}, {127,72,64}, {229,255,128}, {77,138,153}, {213,128,255}, {230,180,172}, {195,217,108}, {57,103,115}, {170,102,204}, {179,140,134}, {103,115,57}, {45,80,89}, {53,32,64}, {128,100,96}, {80,89,45}, {26,46,51}, {164,134,179}, {255,68,0}, {57,64,32}, {191,242,255}, {173,0,217}, {204,54,0}, {206,217,163}, {143,182,191}, {92,0,115}, {166,44,0}, {133,140,105}, {96,121,128}, {61,0,77}, {127,34,0}, {97,102,77}, {0,170,255}, {163,48,191}, {76,20,0}, {73,77,57}, {0,128,191}, {119,35,140}, {51,14,0}, {170,255,0}, {0,102,153}, {149,83,166}, {255,115,64}, {182,242,61}, {0,68,102}, {103,57,115}, {204,92,51}, {134,179,45}, {0,51,77}, {206,163,217}, {166,75,41}, {96,128,32}, {0,34,51}, {109,86,115}, {127,57,32}, {57,77,19}, {64,191,255}, {85,67,89}, {102,46,26}, {149,179,89}, {51,153,204}, {238,0,255}, {76,34,19}, {117,140,70}, {41,124,166}, {143,0,153}, {255,162,128}, {234,255,191}, {32,96,128}, {83,0,89}, {204,129,102}, {88,166,0}, {128,213,255}, {73,19,77}, {166,105,83}, {48,89,0}, {102,170,204}, {247,128,255}, {127,81,64}, {133,204,51}, {83,138,166}, {197,102,204}, {89,57,45}, {99,153,38}, {51,85,102}, {86,45,89}, {64,40,32}, {195,255,128}, {38,64,77}, {251,191,255}, {255,208,191}, {166,217,108}, {182,222,242}, {217,0,202}, {204,167,153}, {39,51,26}, {124,152,166}, {179,0,167}, {166,135,124}, {158,179,134}, {77,94,102}, {115,0,107}, {102,83,77}, {102,255,0}, {48,58,64}, {64,0,60}, {77,62,57}, {51,128,0}, {0,136,255}, {255,64,242}, {255,102,0}, {133,242,61}, {0,109,204}, {153,38,145}, {204,82,0}, {63,115,29}, {0,88,166}, {51,13,48}, {166,66,0}, {35,64,16}, {0,68,128}, {140,70,136}, {127,51,0}, {80,115,57}, {0,48,89}, {153,115,150}, {102,41,0}, {62,89,45}, {0,34,64}, {255,0,204}, {76,31,0}, {184,217,163}, {54,141,217}, {179,0,143}, {51,20,0}, {108,128,96}, {32,83,128}, {140,0,112}, {255,140,64}, {43,51,38}, {128,196,255}, {102,0,82}, {204,112,51}, {58,217,0}, {102,156,204}, {217,54,184}, {166,91,41}, {92,204,51}, {77,117,153}, {255,128,229}, {127,70,32}, {161,255,128}, {51,78,102}, {179,89,161}, {102,56,26}, {113,179,89}, {32,49,64}, {115,57,103}, {76,42,19}, {89,140,70}, {191,225,255}, {64,32,57}, {255,179,128}, {208,255,191}, {153,180,204}, {204,153,194}, {204,143,102}, {22,166,0}, {105,124,140}, {255,0,170}, {166,116,83}, {14,102,0}, {0,102,255}, {204,0,136}, {127,89,64}, {10,77,0}, {0,71,179}, {153,0,102}, {89,62,45}, {7,51,0}, {0,56,140}, {89,0,60}, {64,45,32}, {123,217,108}, {0,36,89}, {64,0,43}, {255,217,191}, {120,153,115}, {0,26,64}, {128,32,96}, {204,173,153}, {80,102,77}, {128,179,255}, {217,108,181}, {166,141,124}, {60,77,57}, {96,134,191}, {153,77,128}, {128,108,96}, {143,191,143}, {77,107,153}, {89,45,74}, {255,136,0}, {48,191,67}, {51,71,102}, {51,26,43}, {204,109,0}, {32,128,45}, {38,54,77}, {255,191,234}, {166,88,0}, {127,255,145}, {26,36,51}, {115,86,105}, {127,68,0}, {51,102,58}, {191,217,255}, {64,48,58}, {102,54,0}, {32,64,36}, {134,152,179}, {255,0,136}, {76,41,0}, {0,255,68}, {86,98,115}, {191,0,102}, {51,27,0}, {0,153,41}, {0,68,255}, {140,0,75}, {255,166,64}, {108,217,137}, {0,58,217}, {102,0,54}, {204,133,51}, {89,179,113}, {0,48,179}, {76,0,41}, {166,108,41}, {70,140,89}, {0,31,115}, {51,0,27}, {127,83,32}, {191,255,208}, {0,24,89}, {242,61,157}, {102,66,26}, {0,230,92}, {0,17,64}, {179,45,116}, {64,41,16}, {0,102,41}, {54,98,217}, {115,29,75}, {255,196,128}, {0,77,31}, {29,52,115}, {255,128,196}, {204,156,102}, {0,51,20}, {128,162,255}, {191,96,147}, {166,127,83}, {128,255,178}, {102,129,204}, {128,64,98}, {127,98,64}, {163,217,184}, {83,105,166}, {191,143,169}, {89,68,45}, {134,179,152}, {64,81,128}, {153,115,135}, {64,49,32}, {105,140,119}, {163,177,217}, {89,67,79}, {255,225,191}, {67,89,76}, {105,115,140}, {255,0,102}, {204,180,153}, {48,64,54}, {67,73,89}, {191,0,77}, {166,146,124}, {0,255,136}, {48,52,64}, {127,0,51}, {128,113,96}, {0,217,116}, {0,34,255}, {102,0,41}, {89,79,67}, {0,179,95}, {0,29,217}, {64,0,26}, {64,56,48}, {0,140,75}, {0,24,179}, {229,57,126}, {255,170,0}, {19,77,50}, {0,14,102}, {166,41,91}, {204,136,0}, {13,51,33}, {0,10,77}, {230,115,161}, {166,111,0}, {108,217,166}, {64,89,255}, {166,83,116}, {127,85,0}, {89,179,137}, {51,71,204}, {102,51,71}, {89,60,0}, {70,140,108}, {38,54,153}, {77,38,54}, {51,34,0}, {51,102,78}, {128,145,255}, {255,191,217}, {255,191,64}, {86,115,101}, {102,116,204}, {255,0,68}, {204,153,51}, {0,255,170}, {77,87,153}, {191,0,51}, {166,124,41}, {0,217,145}, {51,58,102}, {127,0,34}, {115,86,29}, {0,179,119}, {38,43,77}, {102,0,27}, {76,57,19}, {0,128,85}, {26,29,51}, {76,0,20}, {255,213,128}, {0,102,68}, {191,200,255}, {229,57,103}, {204,170,102}, {128,255,212}, {134,140,179}, {166,41,75}, {166,138,83}, {191,255,234}, {86,90,115}, {76,19,34}, {127,106,64}, {38,51,47}, {0,0,153}, {51,13,23}, {89,74,45}, {0,255,204}, {0,0,140}, {255,128,162}, {51,43,26}, {0,217,173}, {0,0,128}, {204,102,129}, {255,234,191}, {0,153,122}, {35,35,140}, {140,70,89}, {191,175,143}, {0,115,92}, {128,128,255}, {51,26,32}, {255,204,0}, {0,89,71}, {89,89,179}, {217,163,177}, {191,153,0}, {0,64,51}, {163,163,217}, {166,124,135}, {127,102,0}, {48,191,163}, {105,105,140}, {128,96,104}, {102,82,0}, {121,242,218}, {57,57,77}, {255,0,34}, {64,51,0}, {83,166,149}, {38,38,51}, {204,0,27}, {229,195,57}, {57,115,103}, {79,70,140}, {153,0,20}, {166,141,41}, {45,89,80}, {51,45,89}, {115,0,15}, {89,76,22}, {32,64,57}, {36,32,64}, {229,57,80}, {51,43,13}, {163,217,206}, {200,191,255}, {166,41,58}, {255,230,128}, {134,179,170}, {14,0,51}, {64,16,22}, {204,184,102}, {105,140,133}, {115,64,255}, {255,128,145}, {166,149,83}, {67,89,85}, {98,54,217}, {178,89,101}};
            assert(n < sizeof(colors));

            ColorMap cmap(n);
            for (size_t i = 0; i < n; ++i)
                cmap[i] = {colors[i][2], colors[i][1], colors[i][0]}; // Colors are in rgb
            return cmap;
        }

        RGBImage colorize(LabelImage const& labelImg, ColorMap const& colorMap)
        {
            RGBImage rgb(labelImg.width(), labelImg.height());

            for (size_t i = 0; i < labelImg.pixels(); ++i)
            {
                Label l = labelImg.atSite(i);
                assert(colorMap.size() > l);

                rgb.atSite(i, 0) = colorMap[l][0];
                rgb.atSite(i, 1) = colorMap[l][1];
                rgb.atSite(i, 2) = colorMap[l][2];
            }

            return rgb;
        }

        LabelImage decolorize(RGBImage const& rgb, ColorMap const& colorMap)
        {
            // Generate lookup-table by colors
            using Color = std::array<unsigned char, 3>;
            boost::unordered_map<Color, size_t> lookup;
            for (size_t i = 0; i < colorMap.size(); ++i)
                lookup[colorMap[i]] = i;

            // Compute label image
            LabelImage labels(rgb.width(), rgb.height());
            for (SiteId s = 0; s < labels.pixels(); ++s)
            {
                Color c = {rgb.atSite(s, 0), rgb.atSite(s, 1), rgb.atSite(s, 2)};
                assert(lookup.count(c) > 0);
                Label l = lookup[c];
                labels.atSite(s) = l;
            }

            return labels;
        }

        RGBImage outline(LabelImage const& labelImg, RGBImage const& colorImg, std::array<unsigned short, 3> const& color)
        {
            assert(labelImg.width() == colorImg.width() && labelImg.height() == colorImg.height());

            RGBImage result = colorImg;
            for (Coord x = 0; x < labelImg.width() - 1; ++x)
            {
                for (Coord y = 0; y < labelImg.height() - 1; ++y)
                {
                    // If label changes
                    if (labelImg.at(x, y) != labelImg.at(x + 1, y) || labelImg.at(x, y) != labelImg.at(x, y + 1))
                    {
                        result.at(x, y, 0) = color[0];
                        result.at(x, y, 1) = color[1];
                        result.at(x, y, 2) = color[2];
                    }
                }
            }

            return result;
        }

        PNGError readPalettePNG(std::string const& file, LabelImage& outImage, ColorMap* pOutColorMap)
        {
            // Open the file
            FILE* fp = fopen(file.c_str(), "rb");
            if (!fp)
            {
                std::cerr << std::strerror(errno) << std::endl;
                return PNGError::CantOpenFile;
            }

            // Check, if it actually is a valid PNG
            uint8_t header[8];
            size_t read = fread(header, 1, 8, fp);
            if (read != 8)
            {
                fclose(fp);
                return PNGError::InvalidFileFormat;
            }
            bool is_png = !png_sig_cmp(header, 0, 8);
            if (!is_png)
            {
                fclose(fp);
                return PNGError::InvalidFileFormat;
            }

            // Setup libPNG
            png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
            if (!png_ptr)
            {
                fclose(fp);
                return PNGError::CantInitReadStruct;
            }

            png_infop info_ptr = png_create_info_struct(png_ptr);
            if (!info_ptr)
            {
                png_destroy_read_struct(&png_ptr, nullptr, nullptr);
                fclose(fp);
                return PNGError::CantInitInfoStruct;
            }

            png_infop end_info = png_create_info_struct(png_ptr);
            if (!end_info)
            {
                png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
                fclose(fp);
                return PNGError::CantInitInfoStruct;
            }

            // Setup return location in case of errors
            if (setjmp(png_jmpbuf(png_ptr)))
            {
                png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
                fclose(fp);
                return PNGError::Critical;
            }

            // Init actual I/O
            png_init_io(png_ptr, fp);
            png_set_sig_bytes(png_ptr, 8); // 8 bytes missing due to PNG validity check in the beginning

            // Read all chunks up to the actual image data
            png_read_info(png_ptr, info_ptr);

            png_uint_32 width, height;
            int bit_depth, color_type, interlace_type, compression_type, filter_method;
            png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, &interlace_type,
                         &compression_type, &filter_method);

            if (color_type != PNG_COLOR_TYPE_PALETTE)
            {
                png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
                fclose(fp);
                return PNGError::NoPalette;
            }
            if (interlace_type != PNG_INTERLACE_NONE)
            {
                png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
                fclose(fp);
                return PNGError::UnsupportedInterlaceType;
            }

            // Read actual image data
            png_bytep* row_pointers = (png_bytepp) png_malloc(png_ptr, sizeof(png_bytep) * height);
            if (!row_pointers)
            {
                png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
                fclose(fp);
                return PNGError::OutOfMemory;
            }
            for (Coord y = 0; y < height; ++y)
                row_pointers[y] = (png_bytep) png_malloc(png_ptr, png_get_rowbytes(png_ptr, info_ptr));
            png_read_image(png_ptr, row_pointers);

            // End reading
            png_read_end(png_ptr, end_info);

            // Palette info
            if (pOutColorMap)
            {
                // Read palette info
                int num_palette;
                png_colorp palette; // Memory will be allocated internally
                png_get_PLTE(png_ptr, info_ptr, &palette, &num_palette);

                pOutColorMap->clear();
                pOutColorMap->reserve(num_palette);
                for (size_t i = 0; i < (size_t) num_palette; ++i)
                {
                    std::array<unsigned char, 3> color;
                    color[0] = palette[i].blue; // Color maps use BGR, not RGB
                    color[1] = palette[i].green;
                    color[2] = palette[i].red;
                    pOutColorMap->push_back(color);
                }
            }

            // Copy data to actual label image
            LabelImage labeling(width, height);
            for (Coord i = 0; i < labeling.height(); ++i)
            {
                for (Coord j = 0; j < labeling.width(); ++j)
                {
                    Label const l = row_pointers[i][j];
                    labeling.at(j, i) = l;
                }
            }
            outImage = labeling;

            // Free all the allocated memory
            for (Coord y = 0; y < height; ++y)
                png_free(png_ptr, row_pointers[y]);
            png_free(png_ptr, row_pointers);
            png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);

            fclose(fp);

            return PNGError::Okay;
        }

        PNGError writePalettePNG(std::string const& file, LabelImage const& labeling, ColorMap const& cmap)
        {
            // Open file
            FILE* fp = fopen(file.c_str(), "wb");
            if (!fp)
            {
                std::cerr << std::strerror(errno) << std::endl;
                return PNGError::CantOpenFile;
            }

            // Create write struct
            png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
            if (!png_ptr)
            {
                fclose(fp);
                return PNGError::CantInitWriteStruct;
            }

            // Create info struct
            png_infop info_ptr = png_create_info_struct(png_ptr);
            if (!info_ptr)
            {
                png_destroy_write_struct(&png_ptr, nullptr);
                fclose(fp);
                return PNGError::CantInitInfoStruct;
            }

            // Setup return location in case of errors
            if (setjmp(png_jmpbuf(png_ptr)))
            {
                png_destroy_write_struct(&png_ptr, &info_ptr);
                fclose(fp);
                return PNGError::Critical;
            }

            // Init I/O
            png_init_io(png_ptr, fp);

            // Set basic information
            png_set_IHDR(png_ptr, info_ptr, labeling.width(), labeling.height(), 8, PNG_COLOR_TYPE_PALETTE,
                         PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

            // Set color palette information
            int num_palette = (int) cmap.size();
            assert(num_palette <= PNG_MAX_PALETTE_LENGTH);
            png_colorp palette = (png_colorp) png_malloc(png_ptr, num_palette * sizeof(png_color));
            for (int p = 0; p < num_palette; p++)
            {
                png_color* col = &palette[p];
                col->blue = cmap[p][0]; // cmap is BGR, not RGB
                col->green = cmap[p][1];
                col->red = cmap[p][2];
            }
            png_set_PLTE(png_ptr, info_ptr, palette, num_palette);

            // Write info
            png_write_info(png_ptr, info_ptr);

            // Write image data
            png_bytep* row_pointers = (png_bytepp) png_malloc(png_ptr, sizeof(png_bytep) * labeling.height());
            if (!row_pointers)
            {
                png_destroy_write_struct(&png_ptr, &info_ptr);
                fclose(fp);
                return PNGError::OutOfMemory;
            }
            for (Coord y = 0; y < labeling.height(); ++y)
                row_pointers[y] = (png_bytep) png_malloc(png_ptr, png_get_rowbytes(png_ptr, info_ptr));
            for (Coord i = 0; i < labeling.height(); ++i)
            {
                for (Coord j = 0; j < labeling.width(); ++j)
                {
                    Label l = labeling.at(j, i);
                    row_pointers[i][j] = (png_byte) l;
                }
            }
            png_write_image(png_ptr, row_pointers);

            // Write the end bit
            png_write_end(png_ptr, info_ptr);

            // Free memory
            png_free(png_ptr, palette);
            for (Coord y = 0; y < labeling.height(); ++y)
                png_free(png_ptr, row_pointers[y]);
            png_free(png_ptr, row_pointers);
            png_destroy_write_struct(&png_ptr, &info_ptr);

            fclose(fp);

            return PNGError::Okay;
        }
    }
}
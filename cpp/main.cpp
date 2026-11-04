#include "audiosr.hpp"
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {
struct Wav { int sr = 0; std::vector<float> data; };
bool read_wav(const char* path, Wav& w) {
    std::ifstream f(path, std::ios::binary); if (!f) return false;
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {});
    if (b.size() < 12 || memcmp(b.data(),"RIFF",4) || memcmp(b.data()+8,"WAVE",4)) return false;
    int ch=1, bits=16, fmt=1; size_t off=0, sz=0, p=12;
    while (p+8<=b.size()) {
        uint32_t n=*reinterpret_cast<const uint32_t*>(&b[p+4]);
        if (!memcmp(&b[p],"fmt ",4) && p+24<=b.size()) { fmt=*reinterpret_cast<const uint16_t*>(&b[p+8]); ch=*reinterpret_cast<const uint16_t*>(&b[p+10]); w.sr=*reinterpret_cast<const int32_t*>(&b[p+12]); bits=*reinterpret_cast<const uint16_t*>(&b[p+22]); }
        else if (!memcmp(&b[p],"data",4)) { off=p+8; sz=n; }
        p += 8+n+(n&1);
    }
    if (!off||!ch||!w.sr) return false;
    size_t frames=sz/(bits/8*ch); w.data.resize(frames);
    for (size_t i=0;i<frames;i++) { double a=0;
        for (int c=0;c<ch;c++) { const uint8_t* q=&b[off+(i*ch+c)*(bits/8)];
            if (fmt==3&&bits==32) { float v; memcpy(&v,q,4); a+=v; }
            else if (bits==16) { int16_t v; memcpy(&v,q,2); a+=v/32768.0; }
            else if (bits==24) { int32_t v=q[0]|(q[1]<<8)|(q[2]<<16); if(v&0x800000)v-=0x1000000; a+=v/8388608.0; }
            else if (bits==32) { int32_t v; memcpy(&v,q,4); a+=v/2147483648.0; }
            else return false;
        }
        w.data[i]=(float)(a/ch);
    }
    return true;
}
void write_wav(const char* path, const std::vector<float>& d, int sr) {
    std::ofstream f(path, std::ios::binary);
    int32_t n=(int32_t)d.size(), br=sr*2, db=n*2, ch=36+db;
    int16_t one=1, two=2, sixteen=16;
    f.write("RIFF",4); f.write((char*)&ch,4); f.write("WAVE",4); f.write("fmt ",4);
    int32_t fmt=16; f.write((char*)&fmt,4); f.write((char*)&one,2); f.write((char*)&one,2);
    f.write((char*)&sr,4); f.write((char*)&br,4); f.write((char*)&two,2); f.write((char*)&sixteen,2);
    f.write("data",4); f.write((char*)&db,4);
    for (float v : d) { int16_t s=(int16_t)((v>1?1:(v<-1?-1:v))*32767); f.write((char*)&s,2); }
}
}
int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr,"usage: %s <models_dir> <input.wav> [output.wav] [--bench]\n",argv[0]); return 1; }
    const std::string models=argv[1], in=argv[2], out=argc>3&&argv[3][0]!='-'?argv[3]:"output.wav";
    bool bench=false; for(int i=3;i<argc;i++) if(!strcmp(argv[i],"--bench")) bench=true;
    Wav w; if(!read_wav(in.c_str(),w)) { fprintf(stderr,"cannot read %s\n",in.c_str()); return 1; }
    audiosr::AudioSR m;
    if(!m.load(models,models+"/mel_filters_256x1025.bin")) { fprintf(stderr,"model load failed\n"); return 1; }
    auto t0=std::chrono::steady_clock::now(); auto y=m.enhance(w.data,w.sr); auto t1=std::chrono::steady_clock::now();
    if(y.empty()) { fprintf(stderr,"inference failed\n"); return 1; }
    double dt=std::chrono::duration<double>(t1-t0).count();
    if(bench) { double s=0; for(int i=0;i<3;i++){auto a=std::chrono::steady_clock::now(); y=m.enhance(w.data,w.sr); auto b=std::chrono::steady_clock::now(); s+=std::chrono::duration<double>(b-a).count();} dt=s/3; }
    write_wav(out.c_str(),y,48000);
    double dur=(double)w.data.size()/w.sr;
    printf("AudioSR: %d Hz -> 48000 Hz, %.2fs audio\n",w.sr,dur);
    printf("  time : %.3fs  RTF %.4f\n",dt,dt/std::max(dur,1e-9));
    printf("  rms  : %.4f / %.4f\n",audiosr::rms(w.data),audiosr::rms(y));
    printf("  output: %s\n",out.c_str()); return 0;
}

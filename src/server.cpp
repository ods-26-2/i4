#include "crow.h"

#include <opencv2/opencv.hpp>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// ---------------------------------------------------------------------
// Memória compartilhada (layout igual ao shm_reader.py / leitor.py)
// ---------------------------------------------------------------------

struct alignas(64) ShmHeader {
    uint32_t magic, width, height, channels;
    std::atomic<uint64_t> seq;
    std::atomic<uint64_t> frame_id;
    uint64_t ts_ms;
};
static_assert(sizeof(ShmHeader) == 64);

class FrameShm {
private:
    std::string nome_;
    int w_, h_, fd_ = -1;
    size_t size_ = 0;
    uint8_t* base_ = nullptr;
    ShmHeader* hdr_ = nullptr;

public:
    FrameShm(const char* nome, int w, int h) : nome_(nome), w_(w), h_(h) {
        size_ = sizeof(ShmHeader) + size_t(w) * h * 3;
        fd_ = shm_open(nome, O_CREAT | O_RDWR, 0666);
        if (fd_ < 0) return;
        if (ftruncate(fd_, size_) < 0) return;
        void* p = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) return;
        base_ = static_cast<uint8_t*>(p);
        hdr_ = reinterpret_cast<ShmHeader*>(base_);
        hdr_->magic = 0x46524D31;  // "FRM1"
        hdr_->width = w; hdr_->height = h; hdr_->channels = 3;
        hdr_->seq.store(0); hdr_->frame_id.store(0);
    }
    ~FrameShm() {
        if (base_) munmap(base_, size_);
        if (fd_ >= 0) { close(fd_); shm_unlink(nome_.c_str()); }
    }

    bool ok() const { return base_ != nullptr; }
    int w() const { return w_; }
    int h() const { return h_; }

    void escrever(const cv::Mat& bgr, uint64_t ts_ms) {
        // bgr precisa ser CV_8UC3 contínuo e do tamanho w_ x h_
        uint64_t s = hdr_->seq.load(std::memory_order_relaxed);
        hdr_->seq.store(s + 1, std::memory_order_relaxed);   // ímpar = escrevendo
        std::atomic_thread_fence(std::memory_order_release);

        std::memcpy(base_ + sizeof(ShmHeader), bgr.data, size_t(w_) * h_ * 3);
        hdr_->ts_ms = ts_ms;
        hdr_->frame_id.fetch_add(1, std::memory_order_relaxed);

        hdr_->seq.store(s + 2, std::memory_order_release);   // par = pronto
    }
};

// ---------------------------------------------------------------------
// Estado da conexão (uma única conexão ativa)
// ---------------------------------------------------------------------

struct EstadoVideo {
    uint64_t frames_recebidos = 0;
    cv::Mat bgr;
    std::unique_ptr<FrameShm> shm;
};

static std::unique_ptr<EstadoVideo> g_estado;
static crow::websocket::connection* g_conn = nullptr;

static void escrever_na_shm(EstadoVideo& st, const cv::Mat& bgr) {
    const int w = bgr.cols, h = bgr.rows;

    // Cria (ou recria, se a resolução mudou) a região de memória compartilhada.
    if (!st.shm || st.shm->w() != w || st.shm->h() != h) {
        st.shm.reset();  // libera o antigo antes de recriar com o mesmo nome
        st.shm = std::make_unique<FrameShm>("/frames_cam", w, h);
        if (!st.shm->ok()) {
            CROW_LOG_ERROR << "Falha ao criar shm";
            st.shm.reset();
            return;
        }
    }

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    st.shm->escrever(bgr, ms);
}

static void consumir_video_jpeg(EstadoVideo& st, const std::string& data) {
    if (data.empty()) return;

    std::vector<uchar> buffer(data.begin(), data.end());
    st.bgr = cv::imdecode(buffer, cv::IMREAD_COLOR);
    if (st.bgr.empty()) {
        CROW_LOG_ERROR << "Falha ao descodificar JPEG recebido";
        return;
    }
    if (!st.bgr.isContinuous()) st.bgr = st.bgr.clone();  // o memcpy da shm exige buffer contínuo

    st.frames_recebidos++;
    escrever_na_shm(st, st.bgr);
}

// ---------------------------------------------------------------------
// App
// ---------------------------------------------------------------------

int main() {
    crow::SimpleApp app;

    CROW_WEBSOCKET_ROUTE(app, "/ws/video")
        .onopen([](crow::websocket::connection& conn) {
            if (g_conn) {
                conn.close("já existe uma conexão de vídeo ativa");
                return;
            }
            g_estado = std::make_unique<EstadoVideo>();
            g_conn = &conn;
            CROW_LOG_INFO << "Conexão de vídeo aberta";
        })
        .onclose([](crow::websocket::connection& conn, const std::string& reason, uint16_t) {
            if (&conn != g_conn) return;  // conexão recusada
            CROW_LOG_INFO << "Conexão de vídeo encerrada (" << reason
                          << "), frames recebidos: " << g_estado->frames_recebidos;
            g_estado.reset();  // destrói a shm (shm_unlink); o ao_vivo.py reabre pelo inode
            g_conn = nullptr;
        })
        .onerror([](crow::websocket::connection&, const std::string& msg) {
            CROW_LOG_ERROR << "Erro no WebSocket de vídeo: " << msg;
        })
        .onmessage([](crow::websocket::connection& conn, const std::string& data, bool is_binary) {
            if (!is_binary || &conn != g_conn) return;
            consumir_video_jpeg(*g_estado, data);
        });

    app.port(18080).run();
    return 0;
}
#include "stonks/binance/signer.h"

#include <array>
#include <cstddef>
#include <stdexcept>

#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>

namespace stonks::binance {

namespace {

constexpr std::string_view unreserved =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";

char hex_digit(int v) { return static_cast<char>(v < 10 ? '0' + v : 'A' + (v - 10)); }

} // namespace

std::string url_encode(std::string_view value)
{
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value) {
        if (unreserved.find(static_cast<char>(c)) != std::string_view::npos) {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('%');
            out.push_back(hex_digit(c >> 4));
            out.push_back(hex_digit(c & 0x0F));
        }
    }
    return out;
}

std::string encode_query(const QueryParams& params)
{
    std::string out;
    bool first = true;
    for (const auto& [key, value] : params) {
        if (!first) { out.push_back('&'); }
        first = false;
        out += key;
        out.push_back('=');
        out += url_encode(value);
    }
    return out;
}

std::string base64_encode(const unsigned char* data, std::size_t len)
{
    if (len == 0) { return {}; }
    // EVP_EncodeBlock writes 4 output chars per 3 input bytes plus a NUL.
    std::string out(4 * ((len + 2) / 3), '\0');
    const int written = EVP_EncodeBlock(reinterpret_cast<unsigned char*>(out.data()),
                                        data, static_cast<int>(len));
    out.resize(static_cast<std::size_t>(written));
    return out;
}

void Ed25519Signer::PKeyDeleter::operator()(void* p) const noexcept
{
    EVP_PKEY_free(static_cast<EVP_PKEY*>(p));
}

Ed25519Signer::Ed25519Signer(std::string_view pem)
{
    std::unique_ptr<BIO, decltype(&BIO_free)> bio{
        BIO_new_mem_buf(pem.data(), static_cast<int>(pem.size())), &BIO_free };
    if (!bio) { throw std::runtime_error{ "Ed25519Signer: BIO allocation failed" }; }

    EVP_PKEY* raw = PEM_read_bio_PrivateKey(bio.get(), nullptr, nullptr, nullptr);
    if (!raw) {
        throw std::runtime_error{ "Ed25519Signer: could not parse Ed25519 private key PEM" };
    }
    m_pkey.reset(raw);

    if (EVP_PKEY_get_base_id(raw) != EVP_PKEY_ED25519) {
        throw std::runtime_error{ "Ed25519Signer: key is not Ed25519" };
    }
}

std::string Ed25519Signer::sign(std::string_view payload) const
{
    std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx{
        EVP_MD_CTX_new(), &EVP_MD_CTX_free };
    if (!ctx) { throw std::runtime_error{ "Ed25519Signer: MD context allocation failed" }; }

    auto* pkey = static_cast<EVP_PKEY*>(m_pkey.get());
    // Ed25519 requires the one-shot EVP_DigestSign API with a null digest.
    if (EVP_DigestSignInit(ctx.get(), nullptr, nullptr, nullptr, pkey) != 1) {
        throw std::runtime_error{ "Ed25519Signer: DigestSignInit failed" };
    }

    const auto* msg = reinterpret_cast<const unsigned char*>(payload.data());
    std::size_t sig_len = 0;
    if (EVP_DigestSign(ctx.get(), nullptr, &sig_len, msg, payload.size()) != 1) {
        throw std::runtime_error{ "Ed25519Signer: signature length probe failed" };
    }
    std::array<unsigned char, 64> sig{};   // Ed25519 signatures are always 64 bytes
    if (sig_len > sig.size()) {
        throw std::runtime_error{ "Ed25519Signer: unexpected signature length" };
    }
    if (EVP_DigestSign(ctx.get(), sig.data(), &sig_len, msg, payload.size()) != 1) {
        throw std::runtime_error{ "Ed25519Signer: signing failed" };
    }
    return base64_encode(sig.data(), sig_len);
}

} // namespace stonks::binance

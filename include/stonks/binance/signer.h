#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace stonks::binance {

// An ordered list of query parameters. Order is preserved from insertion: the
// signed string and the transmitted string are built from the same sequence, so
// the two always agree regardless of ordering (Binance verifies the signature
// over the exact bytes sent).
using QueryParams = std::vector<std::pair<std::string, std::string>>;

// Percent-encode one value per RFC 3986 (unreserved chars A-Za-z0-9-._~ pass
// through; everything else becomes %XX). Applied to every parameter value so a
// symbol/clientOrderId containing a reserved character can never corrupt the
// query or the signature.
std::string url_encode(std::string_view value);

// Render params as `key=url_encode(value)` joined by '&'. Keys are ASCII and
// used verbatim; values are encoded. This is both the string that gets signed
// and (with `&signature=...` appended) the string that gets sent.
std::string encode_query(const QueryParams& params);

// Ed25519 request signer backed by an OpenSSL private key. Construct from a
// PKCS#8 PEM (the format Binance's key generator emits). Signing is one-shot
// (Ed25519 takes no streaming/hash init) and returns the 64-byte signature
// base64-encoded — the representation Binance expects on the `signature` param.
class Ed25519Signer
{
public:
    // Load the private key from PEM text. Throws std::runtime_error if the PEM
    // is unreadable or is not an Ed25519 key.
    explicit Ed25519Signer(std::string_view pem);

    // Sign `payload` (the encoded query string), returning base64(signature).
    std::string sign(std::string_view payload) const;

private:
    struct PKeyDeleter { void operator()(void* p) const noexcept; };
    // EVP_PKEY*, type-erased so the header stays free of <openssl/*.h>.
    std::unique_ptr<void, PKeyDeleter> m_pkey;
};

// Standard base64 (with padding) of arbitrary bytes. Exposed for tests.
std::string base64_encode(const unsigned char* data, std::size_t len);

} // namespace stonks::binance

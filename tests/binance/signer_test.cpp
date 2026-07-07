// Ed25519 signing + query canonicalization. The round-trip verifies against a
// public key derived from the same test key, independent of the signer's path.

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <openssl/evp.h>
#include <openssl/pem.h>

#include "stonks/binance/signer.h"
#include "fake_binance.h"

namespace stonks::binance {
namespace {

using test::kTestPem;

TEST(UrlEncode, LeavesUnreservedAndEncodesReserved)
{
    EXPECT_EQ(url_encode("BTCUSDT"), "BTCUSDT");
    EXPECT_EQ(url_encode("abc-._~"), "abc-._~");
    EXPECT_EQ(url_encode("a b"), "a%20b");
    EXPECT_EQ(url_encode("a+b/c=d&e"), "a%2Bb%2Fc%3Dd%26e");
    EXPECT_EQ(url_encode("stk-c-123-4"), "stk-c-123-4");   // client-id chars pass through
}

TEST(EncodeQuery, PreservesInsertionOrderAndEncodesValues)
{
    const QueryParams p = { { "symbol", "BTCUSDT" }, { "side", "BUY" }, { "note", "a b" } };
    EXPECT_EQ(encode_query(p), "symbol=BTCUSDT&side=BUY&note=a%20b");
}

TEST(Base64, KnownVectors)
{
    const auto enc = [](const std::string& s) {
        return base64_encode(reinterpret_cast<const unsigned char*>(s.data()), s.size());
    };
    EXPECT_EQ(enc("Man"), "TWFu");
    EXPECT_EQ(enc("Ma"), "TWE=");
    EXPECT_EQ(enc("M"), "TQ==");
    EXPECT_EQ(enc(""), "");
}

TEST(Ed25519Signer, RejectsInvalidPem)
{
    EXPECT_THROW(Ed25519Signer{ "not a pem" }, std::runtime_error);
}

TEST(Ed25519Signer, IsDeterministic)
{
    Ed25519Signer signer{ kTestPem };
    const std::string payload = "symbol=BTCUSDT&timestamp=1700000000000";
    EXPECT_EQ(signer.sign(payload), signer.sign(payload));
}

TEST(Ed25519Signer, SignatureVerifiesAgainstPublicKey)
{
    Ed25519Signer signer{ kTestPem };
    const std::string payload = "symbol=BTCUSDT&side=BUY&quantity=0.001&timestamp=1700000000000";
    const std::string sig_b64 = signer.sign(payload);

    // Decode base64 back to the 64 raw signature bytes.
    std::vector<unsigned char> sig(sig_b64.size());
    const int n = EVP_DecodeBlock(sig.data(),
                                  reinterpret_cast<const unsigned char*>(sig_b64.data()),
                                  static_cast<int>(sig_b64.size()));
    ASSERT_GE(n, 64);
    sig.resize(64);

    // Load the keypair (private PEM carries the public half) and verify.
    BIO* bio = BIO_new_mem_buf(kTestPem, -1);
    EVP_PKEY* pkey = PEM_read_bio_PrivateKey(bio, nullptr, nullptr, nullptr);
    BIO_free(bio);
    ASSERT_NE(pkey, nullptr);

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    ASSERT_EQ(EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, pkey), 1);
    const int ok = EVP_DigestVerify(ctx, sig.data(), sig.size(),
                                    reinterpret_cast<const unsigned char*>(payload.data()),
                                    payload.size());
    EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(pkey);
    EXPECT_EQ(ok, 1);
}

} // namespace
} // namespace stonks::binance

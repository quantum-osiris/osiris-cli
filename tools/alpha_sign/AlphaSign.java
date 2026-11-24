import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.security.KeyFactory;
import java.security.KeyPairGenerator;
import java.security.MessageDigest;
import java.security.Signature;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.Base64;

/** Ed25519 init/sign for a request body. The private key stays in the file given on the command line. */
public final class AlphaSign {
    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: AlphaSign init <pkcs8> <pubfile> | AlphaSign sign <pkcs8> <message> <sigfile>");
            System.exit(2);
        }
        if (args[0].equals("init")) {
            if (args.length != 3) {
                System.err.println("usage: AlphaSign init <pkcs8> <pubfile>");
                System.exit(2);
            }
            KeyPairGenerator gen = KeyPairGenerator.getInstance("Ed25519");
            java.security.KeyPair pair = gen.generateKeyPair();
            try (FileOutputStream out = new FileOutputStream(args[1])) {
                out.write(pair.getPrivate().getEncoded());
            }
            byte[] raw = rawPublic(pair.getPublic().getEncoded());
            String text = "pub_hex=" + hex(raw) + "\n"
                    + "fingerprint_sha256=" + hex(MessageDigest.getInstance("SHA-256").digest(raw)) + "\n";
            try (FileOutputStream out = new FileOutputStream(args[2])) {
                out.write(text.getBytes("UTF-8"));
            }
            return;
        }
        if (!args[0].equals("sign") || args.length != 4) {
            System.err.println("usage: AlphaSign sign <pkcs8> <message> <sigfile>");
            System.exit(2);
        }
        byte[] pkcs8 = readAll(args[1]);
        KeyFactory factory = KeyFactory.getInstance("Ed25519");
        java.security.PrivateKey key = factory.generatePrivate(new PKCS8EncodedKeySpec(pkcs8));
        Signature sig = Signature.getInstance("Ed25519");
        sig.initSign(key);
        sig.update(readAll(args[2]));
        byte[] encoded = Base64.getEncoder().encode(sig.sign());
        try (FileOutputStream out = new FileOutputStream(args[3])) {
            out.write(encoded);
        }
    }

    static byte[] rawPublic(byte[] spki) {
        if (spki.length < 32) {
            throw new IllegalArgumentException("public key encoding is short");
        }
        byte[] raw = new byte[32];
        System.arraycopy(spki, spki.length - 32, raw, 0, 32);
        return raw;
    }

    static byte[] readAll(String path) throws Exception {
        try (FileInputStream in = new FileInputStream(path)) {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int n;
            while ((n = in.read(buf)) >= 0) {
                out.write(buf, 0, n);
            }
            return out.toByteArray();
        }
    }

    static String hex(byte[] raw) {
        StringBuilder sb = new StringBuilder(raw.length * 2);
        for (byte b : raw) {
            sb.append(Character.forDigit((b >> 4) & 0xf, 16));
            sb.append(Character.forDigit(b & 0xf, 16));
        }
        return sb.toString();
    }
}

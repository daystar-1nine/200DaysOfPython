# Domain Name System (DNS) Resolution Lifecycle

DNS translates human-readable domain names (such as `example.com`) into numerical IP addresses.
When a client requests a domain, the resolution hierarchy traverses:
1. Local browser cache and operating system resolver cache.
2. Recursive DNS resolver (usually provided by an ISP or public resolver like 8.8.8.8).
3. Root nameservers (directing queries to TLD servers).
4. Top-Level Domain (TLD) nameservers (.com, .org).
5. Authoritative nameservers for the specific domain, which return the canonical A or AAAA records.
DNS caching at every level leverages Time-to-Live (TTL) headers to minimize query latency.
# Requests

[![Version](https://img.shields.io/pypi/v/requests.svg?maxAge=86400)](https://pypi.org/project/requests/)
[![Supported Versions](https://img.shields.io/pypi/pyversions/requests.svg)](https://pypi.org/project/requests)
[![Downloads](https://static.pepy.tech/badge/requests/month)](https://pepy.tech/project/requests)
[![Documentation](https://readthedocs.org/projects/requests/badge/?version=latest)](https://requests.readthedocs.io)

**Requests** is a simple, yet elegant, HTTP library.

```python
>>> import requests
>>> r = requests.get('https://httpbin.org/basic-auth/user/pass', auth=('user', 'pass'))
>>> r.status_code
200
```

Requests allows you to send HTTP/1.1 requests extremely easily. There's no need to manually add query strings to your URLs, or to form-encode your `PUT` & `POST` data.

## Installing Requests and Supported Versions

Requests is available on PyPI:

```console
$ python -m pip install requests
```

Requests officially supports Python 3.10+.

## Supported Features & Best-Practices

Requests is ready for the demands of building robust and reliable HTTP-speaking applications, for the needs of today.

- Keep-Alive & Connection Pooling
- International Domains and URLs
- Sessions with Cookie Persistence
- Browser-style TLS/SSL Verification
- Basic & Digest Authentication
- Automatic Content Decompression and Decoding
- Multi-part File Uploads
- SOCKS Proxy Support
- Streaming Downloads

## Cloning the repository

When cloning the Requests repository, you may need to add the `-c fetch.fsck.badTimezone=ignore` flag to avoid an error about a bad commit timestamp.

```shell
git clone -c fetch.fsck.badTimezone=ignore https://github.com/psf/requests.git
```

---

[![Kenneth Reitz](https://raw.githubusercontent.com/psf/requests/main/ext/kr.png)](https://kennethreitz.org) [![Python Software Foundation](https://raw.githubusercontent.com/psf/requests/main/ext/psf.png)](https://www.python.org/psf)

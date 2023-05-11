#ifndef COMMON_HELPER_PLATFORM_H_
#define COMMON_HELPER_PLATFORM_H_

#include <stdlib.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#endif

inline char* realPath(const char* path, char* resolved_path, size_t resolved_path_max) {
#ifdef _WIN32
    return _fullpath(resolved_path, path, resolved_path_max);
#else
    return realpath(path, resolved_path);
#endif
}

inline int mkDir775(const char* pathname) {
#ifdef _WIN32
    int result = _mkdir(pathname);
    if (result == 0) {
        return _chmod(pathname, _S_IREAD | _S_IWRITE | _S_IEXEC);
    }
    return result;
#else
    return mkdir(pathname, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
}

#endif

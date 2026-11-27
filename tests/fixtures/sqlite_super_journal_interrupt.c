#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int unlink(const char *path) {
    int (*next_unlink)(const char *) = dlsym(RTLD_NEXT, "unlink");
    const char *phase = getenv("BALOOT_TEST_SQLITE_COMMIT_INTERRUPTION");
    const char *name = strrchr(path, '/');
    name = name ? name + 1 : path;
    // Deletion of the super-journal is SQLite's multi-file commit point.
    int commit = phase && strncmp(name, "identity.sqlite3-mj",
                                  sizeof("identity.sqlite3-mj") - 1) == 0;
    if (commit && strcmp(phase, "before") == 0) {
        _exit(81);
    }
    int result = next_unlink(path);
    if (commit && result == 0 && strcmp(phase, "after") == 0) {
        _exit(82);
    }
    return result;
}

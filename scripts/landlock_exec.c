#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/landlock.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <unistd.h>

static void fail(const char *message) {
    perror(message);
    exit(70);
}

static __u64 handled_rights(int abi) {
    __u64 rights = LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_WRITE_FILE |
                   LANDLOCK_ACCESS_FS_READ_FILE | LANDLOCK_ACCESS_FS_READ_DIR |
                   LANDLOCK_ACCESS_FS_REMOVE_DIR | LANDLOCK_ACCESS_FS_REMOVE_FILE |
                   LANDLOCK_ACCESS_FS_MAKE_CHAR | LANDLOCK_ACCESS_FS_MAKE_DIR |
                   LANDLOCK_ACCESS_FS_MAKE_REG | LANDLOCK_ACCESS_FS_MAKE_SOCK |
                   LANDLOCK_ACCESS_FS_MAKE_FIFO | LANDLOCK_ACCESS_FS_MAKE_BLOCK |
                   LANDLOCK_ACCESS_FS_MAKE_SYM;
    if (abi >= 2) rights |= LANDLOCK_ACCESS_FS_REFER;
    if (abi >= 3) rights |= LANDLOCK_ACCESS_FS_TRUNCATE;
    return rights;
}

static void add_path_rule(int ruleset_fd, const char *path, __u64 rights) {
    int path_fd = open(path, O_PATH | O_CLOEXEC);
    if (path_fd < 0) fail(path);
    struct stat metadata;
    if (fstat(path_fd, &metadata) < 0) fail("fstat");
    if (!S_ISDIR(metadata.st_mode)) {
        rights &= LANDLOCK_ACCESS_FS_EXECUTE | LANDLOCK_ACCESS_FS_READ_FILE |
                  LANDLOCK_ACCESS_FS_WRITE_FILE | LANDLOCK_ACCESS_FS_TRUNCATE;
    }
    struct landlock_path_beneath_attr rule = {
        .allowed_access = rights,
        .parent_fd = path_fd,
    };
    if (syscall(SYS_landlock_add_rule, ruleset_fd, LANDLOCK_RULE_PATH_BENEATH, &rule, 0) < 0)
        fail("landlock_add_rule");
    if (close(path_fd) < 0) fail("close");
}

static void restrict_socket_families(void) {
#if defined(__aarch64__)
    const unsigned int audit_arch = AUDIT_ARCH_AARCH64;
#elif defined(__x86_64__)
    const unsigned int audit_arch = AUDIT_ARCH_X86_64;
#else
#error "landlock launcher requires an audited seccomp architecture"
#endif
    struct sock_filter filter[] = {
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, audit_arch, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_socket, 0, 3),
        BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, args[0])),
        BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AF_UNIX, 1, 0),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (EACCES & SECCOMP_RET_DATA)),
        BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
    };
    struct sock_fprog program = {
        .len = (unsigned short)(sizeof(filter) / sizeof(filter[0])),
        .filter = filter,
    };
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) < 0)
        fail("seccomp socket filter");
}

int main(int argc, char **argv) {
    int abi = (int)syscall(SYS_landlock_create_ruleset, NULL, 0,
                           LANDLOCK_CREATE_RULESET_VERSION);
    if (abi < 4) {
        errno = ENOPROTOOPT;
        fail("landlock ABI lacks network isolation");
    }
    __u64 handled = handled_rights(abi);
    struct landlock_ruleset_attr ruleset = {
        .handled_access_fs = handled,
        .handled_access_net =
            LANDLOCK_ACCESS_NET_BIND_TCP | LANDLOCK_ACCESS_NET_CONNECT_TCP,
    };
    int ruleset_fd = (int)syscall(SYS_landlock_create_ruleset, &ruleset, sizeof(ruleset), 0);
    if (ruleset_fd < 0) fail("landlock_create_ruleset");

    int index = 1;
    while (index < argc && strcmp(argv[index], "--") != 0) {
        bool writable = strcmp(argv[index], "--rw") == 0;
        bool traverse = strcmp(argv[index], "--traverse") == 0;
        if (!writable && !traverse && strcmp(argv[index], "--ro") != 0) {
            errno = EINVAL;
            fail("landlock option");
        }
        if (++index >= argc) {
            errno = EINVAL;
            fail("landlock path");
        }
        __u64 allowed = traverse
                            ? LANDLOCK_ACCESS_FS_EXECUTE
                            : writable
                                  ? handled
                                  : LANDLOCK_ACCESS_FS_EXECUTE |
                                        LANDLOCK_ACCESS_FS_READ_FILE |
                                        LANDLOCK_ACCESS_FS_READ_DIR;
        add_path_rule(ruleset_fd, argv[index++], allowed);
    }
    if (index >= argc || ++index >= argc) {
        errno = EINVAL;
        fail("landlock command");
    }
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) < 0) fail("PR_SET_NO_NEW_PRIVS");
    restrict_socket_families();
    if (syscall(SYS_landlock_restrict_self, ruleset_fd, 0) < 0)
        fail("landlock_restrict_self");
    if (close(ruleset_fd) < 0) fail("close");
    execvp(argv[index], &argv[index]);
    fail("landlock exec");
    return 70;
}

#ifndef _LOCKED_PERSISTENT_ARCHIVE_HPP_
#define _LOCKED_PERSISTENT_ARCHIVE_HPP_

#include <string>
#include <vector>
#include <functional>

template <typename T>
class LockedPersistentArchive {
    public:
        typedef std::function<void(T&)> ModifyFn;

        explicit LockedPersistentArchive(const std::string& archiveFilename);

        void lockedModify(ModifyFn f) const;
        T lockedRead() const;

    protected:
        std::string _archiveFilename;
};

template <typename T>
class LockedPersistentVector : public LockedPersistentArchive<std::vector<T>> {
    public:
        explicit LockedPersistentVector(const std::string& archiveFilename)
            : LockedPersistentArchive(archiveFilename) { }

        void lockedAppend(const T& elem);
};



#endif 

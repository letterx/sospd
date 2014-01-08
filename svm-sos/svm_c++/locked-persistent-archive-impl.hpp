#ifndef _LOCKED_PERSISTENT_ARCHIVE_UTIL_HPP_
#define _LOCKED_PERSISTENT_ARCHIVE_UTIL_HPP_

#include "util/lockedPersistentArchive.hpp"

template <typename T>
LockedPersistentArchive<T>::LockedPersistentArchive(const std::string& archiveFilename)
    : _archiveFilename(archiveFilename) { }


template <typename T>
void LockedPersistentArchive<T>::lockedModify(ModifyFn f) const {
    boost::interprocess::file_lock flock(fname.c_str());
    {
        boost::interprocess::scoped_lock<boost::interprocess::file_lock> l(flock);

        T value;
        try {
            std::ifstream ifs(fname);
            boost::archive::text_iarchive iar(ifs);
            iar & value;
            ifs.close();
        } catch (std::exception& e) { stats_list = std::vector<TestStats>(); }

        f(value);

        std::ofstream ofs(fname, std::ios_base::trunc | std::ios_base::out);
        boost::archive::text_oarchive oar(ofs);
        oar & value;
    }
}

template <typename T>
T LockedPersistentArchive<T>::lockedRead() const {
    std::ifstream ifs(fname);
    TestStats stats;
    std::vector<TestStats> stats_list;

    boost::archive::text_iarchive ar(ifs);
    ar & stats_list;

    return stats_list;
}

template <typename T>
void LockedPersistentVector<T>::lockedAppend(const T& elem) {
    lockedModify([&](std::vector<T>& vec) { vec.push_back(elem); });
}


#endif

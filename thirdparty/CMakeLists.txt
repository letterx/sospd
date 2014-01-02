###
### Download and extract QPBO
### 

if (NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/QPBO)
    file(DOWNLOAD http://pub.ist.ac.at/~vnk/software/QPBO-v1.3.src.tar.gz 
        ${CMAKE_CURRENT_SOURCE_DIR}/QPBO-v1.3.src.tar.gz)
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf 
        ${CMAKE_CURRENT_SOURCE_DIR}/QPBO-v1.3.src.tar.gz
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    # Copy our version of instances.inc
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/instances.inc
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/QPBO-v1.3.src/)
    file(RENAME ${CMAKE_CURRENT_SOURCE_DIR}/QPBO-v1.3.src 
        ${CMAKE_CURRENT_SOURCE_DIR}/QPBO)
    file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/QPBO-v1.3.src.tar.gz)
endif()

#ifndef _SOS_COMMON_HPP_
#define _SOS_COMMON_HPP_

#include <stdint.h>
#include <limits>
#include <vector>
#include <memory>
#include <assert.h>

#ifndef DNO_ASSERT
#define ASSERT(cond) assert(cond)
#else
#define ASSERT(cond) (void)0
#endif

#endif

#pragma once

namespace erl::sdf_mapping {

    extern bool initialized;

    /**
     * @brief Initialize the library.
     */
    bool
    Init();

    inline const static bool kAutoInitialized = Init();
}

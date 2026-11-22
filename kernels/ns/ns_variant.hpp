// ns_variant.hpp: for a silo variant (kernels/<kind>/<arch>/silo/variants/<name>/, CONTRIBUTING rule 6) - the default
// silo's op of the same name, for the shapes the variant's own op does not cover.
//
// An engine binds a variant's op and its `_supported` check in place of the default silo's, so a variant whose op covers
// less than the default's answers `_supported` for both (its own cover, or the default's) and hands the rest to the
// default's op through this: the default silo is opened globally before any variant (the variant locally, linked
// -Bsymbolic), so the process's global scope finds the default's. Null when there is none (the engine's shared kernel
// then: return the default's `_supported` answer, which is 0 too).
#pragma once
#include <dlfcn.h>

template <class F>
F ns_silo_default(const char* name) {
    return reinterpret_cast<F>(dlsym(RTLD_DEFAULT, name));
}

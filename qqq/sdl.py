import sdl2 as sd2
import sdl2.ext as sdx


def mk_sdl_surface(w, h, depth=32):
    sp = sd2.SDL_CreateRGBSurface(0, w, h, depth,
                                  0x000000ff, 0x0000ff00,
                                  0x00ff0000, 0xff000000)
    if not sp:
        print(sd2.SDL_GetError())
        raise ValueError('null pointer to surface returned')
    else:
        return sp.contents




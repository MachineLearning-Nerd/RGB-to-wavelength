import numpy as np
import matplotlib.pyplot as plt
import cv2
def wavelength_to_rgb(wavelength, gamma=0.8):

    '''This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 1
    G *= 1
    B *= 1
    return (R, G, B)

if __name__ == "__main__":
    image = np.zeros([200 , 371, 3])
    for j in range(200):
        for i in range(380,750):
            (R,G,B) = wavelength_to_rgb(i)
            i = i-380
            image[j][i][0] = R * 255
            image[j][i][1] = G * 255
            image[j][i][2] = B * 255
    plt.imshow(image.astype(np.uint8))
    plt.show()

    image = cv2.imread('input.png')
    # Intensity is the average of all the three elements
    temp = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    plt.imshow(image)
    plt.show()

    # 0th elemwnt is teh hue
    h = image[:,:,0]
    print(np.unique(h))

    I = (temp[:,:,1] + temp[:,:,2] + temp[:,:,0])/(255*3)

    # This is the variable to multiply with the matrix.
    mul = ((470-700)/120)
    L = np.array(h)
    L =  mul * L + 700

    # reshape and sort the wavelength to plot the graph
    temp_L = np.reshape(L,L.shape[0]*L.shape[1])
    sort_L = sorted(temp_L)
    index_sort_L = np.argsort(temp_L)

    temp_I = np.reshape(I, I.shape[0]*I.shape[1])
    sort_I = temp_I[index_sort_L]

    # This is to findout the unique and maximum Intensity for the given
    # value of the wavelength{}
    u_wavelength = np.unique(sort_L)
    print(u_wavelength)
    max_intensity = []
    for i in u_wavelength:
        itemindex = np.where(sort_L==i)
        #maxintensity = max(sort_I[itemindex])
        maxintensity = np.mean(sort_I[itemindex])

        max_intensity.append(maxintensity)

    barlist=plt.bar(u_wavelength, max_intensity)
    for i , wave_len in enumerate(u_wavelength):
        RGBcolors = wavelength_to_rgb(int(wave_len))
        barlist[i].set_color(RGBcolors)
    # Add title and axis names
    plt.title('Specturm of the Colors')
    plt.xlabel('Wavelength')
    plt.ylabel('Intensity')
    plt.xlim(350,720)
    plt.ylim(0,np.max(max_intensity))
    plt.show()

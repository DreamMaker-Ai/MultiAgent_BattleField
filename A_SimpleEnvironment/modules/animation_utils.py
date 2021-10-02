from tensorflow.keras.models import Model, clone_model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_engage_image(env):
    red_matrix = np.zeros((env.grid_size, env.grid_size))
    blue_matrix = np.zeros((env.grid_size, env.grid_size))

    for j in range(env.num_red):
        if env.red.alive[j]:
            red_matrix[env.red.pos[j][0], env.red.pos[j][1]] += env.red.force[j]

    for j in range(env.num_blue):
        if env.blue.alive[j]:
            blue_matrix[env.blue.pos[j][0], env.blue.pos[j][1]] += env.blue.force[j]

    engage_matrix = red_matrix - blue_matrix
    engage_matrix = np.expand_dims(engage_matrix, axis=-1)
    return engage_matrix


def make_engagement_anim(ims, results_path, i, pos):
    eng_ims = []
    fig = plt.figure()
    boundary = np.max([np.abs(np.min(ims)), np.abs(np.min(ims))])
    for s, im in enumerate(ims):
        img = plt.imshow(im, vmin=-boundary, vmax=boundary, animated=True,
                         cmap='bwr')
        txt1 = plt.text(-0.5, 0.0, ("red_0.x=" + str(pos[s][0])), size=10, color="green")
        txt2 = plt.text(-0.5, 0.5, ("red_0.y=" + str(pos[s][1])), size=10, color="green")
        # plt.title(f'red_0.pos=({pos[s][0]}, {pos[s][1]})')
        plt.tick_params(labelbottom=False, bottom=False)
        plt.tick_params(labelleft=False, left=False)
        eng_ims.append([img] + [txt1] + [txt2])
    anim = animation.ArtistAnimation(fig, eng_ims, interval=500, blit=True, repeat_delay=3000)
    anim_name = results_path + 'anim_' + str(i)
    anim.save(anim_name + '.gif', writer='imagemagick')
    anim.save(anim_name + '.mp4', writer='ffmpeg')
    plt.clf()
    plt.cla()
    plt.close(fig)

    return anim_name + '.mp4'


def feature_map_model(trainer, ixs):
    model = trainer.get_policy().model.base_model
    outputs = [model.layers[i].output for i in ixs]
    model = Model(inputs=model.inputs, outputs=outputs)

    return clone_model(model)


def get_feature_map_image(model, obs):
    figs = []
    if 'red_0' in obs:
        layer_outs = model.predict(np.expand_dims(obs['red_0'], axis=0))
        if len(layer_outs) > 1:
            for layer_out in layer_outs:
                fig = np.squeeze(layer_out, axis=0)
                figs.append(fig)
        else:
            fig = layer_outs[0]
            figs.append(fig)
    else:
        for i in range(len(model.outputs)):
            fig = np.zeros(model.output_shape[i][1:3])
            figs.append(fig)
    return figs


def make_feature_map_anim(maps, results_path, i, ixs):
    # maps = np.array(maps)  # (23,2,8,8,16)
    # maps = np.transpose(maps, axes=(1, 0, 2, 3, 4))  # (2,23,8,8,16)
    map_list = []
    for i in range(len(ixs)):
        map = []
        for l in range(len(maps)):
            map.append(maps[l][i])
        map_list.append(map)

    for i in range(len(map_list)):  # Layer output
        ims = []
        features = np.array(map_list[i])  # (23,8,8,16)
        seq_len = features.shape[-1]
        hight = np.int(np.ceil(seq_len ** .5))
        width = hight
        # fig, ax = plt.subplots(width, hight, figsize=(5, 5))
        fig, ax = plt.subplots(width, hight, tight_layout=True)
        fig.tight_layout()
        for j in range(features.shape[0]):  # Sequence
            feature = features[j]  # (8,8,16)
            no_feature = np.ones(feature.shape[0:2])  # Blank subplot
            img = []
            k = 0
            #vmin = np.mean(feature)
            #vmax = np.max(feature)
            vmin = 0
            vmax = (np.max(feature) + np.mean(feature)) / 2
            for w in range(width):  # feature map (channel)
                for h in range(hight):
                    if k < seq_len:
                        """ feature map 有り """
                        ax[(w, h)].tick_params(labelbottom=False, bottom=False)
                        ax[(w, h)].tick_params(labelleft=False, left=False)
                        im = ax[(w, h)].imshow(feature[:, :, k], vmin=vmin, vmax=vmax,
                                               animated=True, cmap='Greys')
                        img += [im]
                        k += 1
                    else:
                        """ feature map無し """
                        ax[(w, h)].tick_params(labelbottom=False, bottom=False)
                        ax[(w, h)].tick_params(labelleft=False, left=False)
                        im = ax[(w, h)].imshow(no_feature, vmin=0, vmax=1,
                                               animated=True, cmap='Greys')

            ims.append(img)
        anim = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=3000)
        if i % 2 == 0:
            anim_name = results_path + 'policy_feature_anim_layer_' + str(ixs[i])
        elif i % 2 == 1:
            anim_name = results_path + 'value_feature_anim_layer_' + str(ixs[i])
        anim.save(anim_name + '.gif', writer='imagemagick')
        anim.save(anim_name + '.mp4', writer='ffmpeg')
        plt.clf()
        plt.close()

        return anim_name + '.mp4'


def get_cumulative_force(force):
    c_force = np.sum(force)
    return c_force


def get_force_image(env):
    red_cumulative_force = get_cumulative_force(env.red.force)
    blue_cumulative_force = get_cumulative_force(env.blue.force)
    return red_cumulative_force, blue_cumulative_force


def make_forces_anim(r_forces, b_forces, results_path, i):
    ims = []
    fig = plt.figure()

    for r_force, b_force in zip(r_forces, b_forces):
        im1 = plt.scatter(['total red force'], r_force, color='r', s=500, marker='^')
        im2 = plt.scatter(['total blue force'], b_force, color='b', s=500, marker='^')
        ims.append([im1] + [im2])

    anim = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=3000)
    anim_name = results_path + 'force_anim_' + str(i)
    anim.save(anim_name + '.gif', writer='imagemagick')
    anim.save(anim_name + '.mp4', writer='ffmpeg')
    plt.clf()
    plt.close()

    return anim_name + '.mp4'
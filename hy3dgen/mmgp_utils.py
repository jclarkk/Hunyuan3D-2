def replace_property_getter(instance, property_name, new_getter):
    # Get the original class and property
    original_class = type(instance)
    original_property = getattr(original_class, property_name)

    # Create a custom subclass for this instance
    custom_class = type(f'Custom{original_class.__name__}', (original_class,), {})

    # Create a new property with the new getter but same setter
    new_property = property(new_getter, original_property.fset)
    setattr(custom_class, property_name, new_property)

    # Change the instance's class
    instance.__class__ = custom_class

    return instance
